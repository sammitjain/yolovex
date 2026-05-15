"""Derived block-level specs.

Builds a per-instance graph for every block in the model, deduplicates
identical graphs into a shared spec (so 7 Conv blocks share one spec entry),
and produces multi-fidelity views by aggregating fx graph nodes at successive
depths of the module hierarchy. **No hand-coded templates — everything is
derived from torch.fx.**

The aggregation rule:
  - call_module nodes whose path is deeper than `depth` collapse into the
    visual node at path[:depth]
  - call_function / call_method nodes (cat, chunk, add, etc.) are structural
    and stay individual at every level
  - placeholder / output stay individual

So fidelity-level == module-hierarchy-depth, not a separately authored
template. Two same-class blocks with different internal modules (e.g.
C3k2 with c3k=False vs c3k=True) hash to different specs; instances merely
link to a spec id and add per-instance shape annotations.

Outputs:
  out/specs/library.json          full deduplicated library
  out/specs/sample/<idx>_L<n>.mmd mermaid for one block at each level

Run:
  uv run python -m yolovex.block_spec --demo-block 9
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from .build_assets_l2 import _preprocess_for_raw_forward
from .introspect import _shape_of, capture_block_input_shapes, structural_signature
from .model import DEFAULT_WEIGHTS, get_blocks, load_model
from .topology import _resolve_sources

# Ultralytics' C2f (parent of C3k2) is the only block in YOLO26's stem that
# fx can't trace as-written, because its forward extends a list with a
# generator expression — fx loses sight of the per-iteration ops inside the
# generator frame. We provide a drop-in replacement that uses an explicit
# for-loop instead, which fx unrolls cleanly over the (statically-sized)
# ModuleList. See _trace_with_patches below.
try:
    from ultralytics.nn.modules.block import C2f as _C2f
except Exception:   # pragma: no cover — Ultralytics is a hard dep at runtime
    _C2f = None

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    name: str               # fx node name (unique per graph)
    op: str                 # placeholder|call_module|call_function|call_method|get_attr|output
    target: str             # dotted module path or function name
    path: list[str]         # raw module-hierarchy lineage (preserved for debugging)
    visible_path: list[str] # container-transparent lineage (used for aggregation)
    target_class: Optional[str]  # for call_module: class name of the resolved submodule
    args: Any                    # _arg_repr-normalised


@dataclass
class BlockGraph:
    class_name: str
    signature: str
    nodes: list[GraphNode]
    edges: list[tuple[str, str]]
    forward_source: str
    derivation_method: str       # 'fx' | 'failed'
    derivation_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Path derivation — the core of multi-fidelity aggregation
# ---------------------------------------------------------------------------

def derive_path(node: torch.fx.Node) -> list[str]:
    """Module-hierarchy lineage. Drives all fidelity aggregation."""
    if node.op == "placeholder" or node.op == "output":
        return ["_io", node.name]
    if node.op == "call_module":
        return str(node.target).split(".")
    if node.op == "call_function":
        fn = getattr(node.target, "__name__", None) or str(node.target)
        return ["_op", fn, node.name]   # node.name keeps repeated calls distinct
    if node.op == "call_method":
        return ["_op", str(node.target), node.name]
    if node.op == "get_attr":
        return ["_attr"] + str(node.target).split(".")
    return ["_unknown", node.name]


def resolve_target_class(module: torch.nn.Module, target: str) -> Optional[str]:
    """Walk a dotted target ('cv1.conv', 'm.0') and return its class name."""
    if not target:
        return None
    obj = module
    try:
        for part in target.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return type(obj).__name__
    except (AttributeError, IndexError, TypeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Visible-path derivation — container-transparent paths
# ---------------------------------------------------------------------------
#
# The "visible" path is the same as the raw module path, except:
#   - segments whose corresponding module is an nn.Sequential or nn.ModuleList
#     are SKIPPED (the container itself isn't an architectural unit)
#   - when a kept segment immediately follows a skipped container, the kept
#     segment is the container's INDEX into that container. We suffix it with
#     the child's class name (e.g. "0_Bottleneck") so siblings inside the
#     same container stay distinguishable.
#
# This lets the depth dial mean "n visible levels deep" rather than "n raw
# levels deep" — matches the user's L1/L2/L3 mental model regardless of how
# many container layers a particular block uses to wrap its components.

_CONTAINER_TYPES = (torch.nn.Sequential, torch.nn.ModuleList)


def compute_visible_path_for_module_target(target: str, root_module: torch.nn.Module) -> list[str]:
    if not target:
        return []
    parts = target.split(".")
    visible: list[str] = []
    cur = root_module
    parent_was_container = False
    for part in parts:
        try:
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        except (AttributeError, IndexError, TypeError, KeyError):
            # Bail out: keep raw segment, reset container flag
            visible.append(part)
            parent_was_container = False
            continue
        is_container = isinstance(cur, _CONTAINER_TYPES)
        if is_container:
            parent_was_container = True
            continue
        if parent_was_container:
            cls = type(cur).__name__
            visible.append(f"{part}_{cls}")
        else:
            visible.append(part)
        parent_was_container = False
    return visible


def compute_visible_path(node: torch.fx.Node, root_module: torch.nn.Module) -> list[str]:
    """Container-transparent path used to drive fidelity aggregation.

    For modules: walk node.target with container transparency.
    For ops (call_function / call_method): use the deepest module in
      node.meta['nn_module_stack'] as the enclosing context. Append
      ['_op', node.name] so each op stays a unique entry.
    For I/O and get_attr: use synthetic prefixes (_io / _attr).
    """
    if node.op in ("placeholder", "output"):
        return ["_io", node.name]
    if node.op == "get_attr":
        return ["_attr"] + str(node.target).split(".")
    if node.op == "call_module":
        # nn_module_stack has the call hierarchy. The LAST entry is the called
        # module itself; the entry BEFORE it is the call site. For non-shared
        # modules these match the target path; for shared modules (one Python
        # object reachable from multiple parents — e.g. Ultralytics' single
        # SiLU instance reused across every Conv block) they diverge, and the
        # call site is what we want for visualization.
        stack = node.meta.get("nn_module_stack", {}) or {}
        target_str = str(node.target)
        leaf = target_str.rsplit(".", 1)[-1] if "." in target_str else target_str
        if stack:
            qnames = list(stack.keys())
            # Strip any '@n' suffixes fx adds for repeated calls.
            qnames_clean = [q.split("@", 1)[0] for q in qnames]
            if len(qnames_clean) >= 2:
                call_site = qnames_clean[-2]
                parent_visible = compute_visible_path_for_module_target(call_site, root_module)
                return parent_visible + [leaf]
            # Single-entry stack: the called module is at the top level
            return [leaf]
        return compute_visible_path_for_module_target(target_str, root_module)
    if node.op in ("call_function", "call_method"):
        stack = node.meta.get("nn_module_stack", {}) or {}
        if stack:
            # nn_module_stack is an OrderedDict keyed by qualified name from
            # shallowest to deepest. Use the deepest entry as the op's home.
            deepest = list(stack.keys())[-1]
            if deepest:
                mod_visible = compute_visible_path_for_module_target(deepest, root_module)
                return mod_visible + ["_op", node.name]
        return ["_op", node.name]
    return ["_op", node.name]


def build_path_class_map(root_module: torch.nn.Module) -> dict[str, str]:
    """Map every visible_path prefix to the class of the module it resolves to.

    Used by the viewer to label aggregated groups with the right class name
    (e.g. group 'cv1' -> 'Conv', group '0_Bottleneck' -> 'Bottleneck').
    """
    result: dict[str, str] = {}

    def walk(mod: torch.nn.Module, visible_prefix: tuple) -> None:
        if isinstance(mod, _CONTAINER_TYPES):
            # Container: its children appear at OUR visible_prefix level.
            for i, sub in enumerate(mod):
                if isinstance(sub, _CONTAINER_TYPES):
                    walk(sub, visible_prefix)  # nested container — fully transparent
                else:
                    cls = type(sub).__name__
                    seg = f"{i}_{cls}"
                    new_prefix = visible_prefix + (seg,)
                    result["/".join(new_prefix)] = cls
                    walk(sub, new_prefix)
            return
        for name, child in mod.named_children():
            if isinstance(child, _CONTAINER_TYPES):
                walk(child, visible_prefix)  # transparent — don't append name
            else:
                cls = type(child).__name__
                new_prefix = visible_prefix + (name,)
                result["/".join(new_prefix)] = cls
                walk(child, new_prefix)

    walk(root_module, ())
    return result


# ---------------------------------------------------------------------------
# fx → BlockGraph
# ---------------------------------------------------------------------------

def _arg_repr(a) -> Any:
    if isinstance(a, torch.fx.Node):
        return a.name
    if isinstance(a, (list, tuple)):
        return [_arg_repr(x) for x in a]
    if isinstance(a, dict):
        return {k: _arg_repr(v) for k, v in a.items()}
    if isinstance(a, (int, float, str, bool)) or a is None:
        return a
    return repr(a)


def _collect_refs(arg, refs: set):
    if isinstance(arg, torch.fx.Node):
        refs.add(arg.name)
    elif isinstance(arg, (list, tuple)):
        for x in arg:
            _collect_refs(x, refs)
    elif isinstance(arg, dict):
        for v in arg.values():
            _collect_refs(v, refs)


def _traceable_c2f_forward(self, x):
    """fx-friendly rewrite of C2f.forward.

    The original is:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    The generator expression hides each m(y[-1]) call inside its own frame so
    fx's tracer never sees the tensor op proxy chain. Rewriting as an explicit
    for-loop over `self.m` (a ModuleList of known length) lets fx unroll the
    iteration statically — every m(y[-1]) becomes a normal call_module node
    in the captured graph. Output graph is semantically identical.
    """
    parts = self.cv1(x).chunk(2, 1)
    y = [parts[0], parts[1]]
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def _trace_with_patches(module):
    """torch.fx.symbolic_trace with per-class forward patches applied.

    fx reads `type(root).forward`, so to install a traceable forward we
    construct a one-off subclass with the override and temporarily re-class
    the instance. The original class is restored in a finally block, so the
    swap is invisible to anything outside this call.
    """
    cls = type(module)
    if _C2f is not None and isinstance(module, _C2f):
        patched = type(
            f"_Traceable_{cls.__name__}",
            (cls,),
            {"forward": _traceable_c2f_forward},
        )
        module.__class__ = patched
        try:
            return torch.fx.symbolic_trace(module)
        finally:
            module.__class__ = cls
    return torch.fx.symbolic_trace(module)


def derive_block_graph(module, input_shape) -> tuple[BlockGraph, dict[str, Any]]:
    """Returns (graph, per-fx-node shape map) for one block instance."""
    cls = type(module).__name__
    sig = structural_signature(module, depth=1)
    src = inspect.getsource(module.__class__.forward)

    if not isinstance(input_shape, tuple) or (input_shape and input_shape[0] == "list"):
        return (
            BlockGraph(
                class_name=cls, signature=sig, nodes=[], edges=[],
                forward_source=src, derivation_method="failed",
                derivation_error=f"non-tensor input ({input_shape})",
            ),
            {},
        )

    try:
        gm = _trace_with_patches(module)
    except Exception as e:
        return (
            BlockGraph(
                class_name=cls, signature=sig, nodes=[], edges=[],
                forward_source=src, derivation_method="failed",
                derivation_error=f"symbolic_trace: {e.__class__.__name__}: {e}",
            ),
            {},
        )

    shapes_by_node: dict[str, Any] = {}
    try:
        ShapeProp(gm).propagate(torch.zeros(*input_shape))
        for n in gm.graph.nodes:
            shapes_by_node[n.name] = _shape_of(n)
    except Exception:
        pass

    nodes: list[GraphNode] = []
    edges: list[tuple[str, str]] = []
    for n in gm.graph.nodes:
        if isinstance(n.target, str):
            target_str = n.target
        else:
            target_str = (
                getattr(n.target, "__qualname__", None)
                or getattr(n.target, "__name__", None)
                or repr(n.target)
            )
        nodes.append(GraphNode(
            name=n.name,
            op=n.op,
            target=target_str,
            path=derive_path(n),
            visible_path=compute_visible_path(n, module),
            target_class=resolve_target_class(module, n.target) if n.op == "call_module" else None,
            args=_arg_repr(n.args),
        ))
        refs: set[str] = set()
        for a in n.args:
            _collect_refs(a, refs)
        for v in n.kwargs.values():
            _collect_refs(v, refs)
        for src_name in refs:
            edges.append((src_name, n.name))

    return (
        BlockGraph(
            class_name=cls, signature=sig, nodes=nodes, edges=edges,
            forward_source=src, derivation_method="fx",
        ),
        shapes_by_node,
    )


# ---------------------------------------------------------------------------
# Topology hash — structural identity for spec deduplication
# ---------------------------------------------------------------------------

def _normalize_for_hash(args):
    if isinstance(args, list):
        return tuple(_normalize_for_hash(a) for a in args)
    if isinstance(args, tuple):
        return tuple(_normalize_for_hash(a) for a in args)
    if isinstance(args, dict):
        return tuple(sorted((k, _normalize_for_hash(v)) for k, v in args.items()))
    return args


def graph_topology_hash(g: BlockGraph) -> str:
    """Hash structural identity: class + signature + (op, target_class, arg-topology)
    sequence + edges. Excludes weights, shapes, parameter values."""
    if g.derivation_method != "fx":
        h = hashlib.sha256(
            f"{g.class_name}|{g.signature}|FAILED:{g.derivation_error}".encode()
        ).hexdigest()[:16]
        return f"fail_{h}"

    parts: list[str] = [g.class_name, g.signature]
    for n in g.nodes:
        parts.append(repr((
            n.op,
            n.target_class or n.target,
            tuple(n.visible_path),
            _normalize_for_hash(n.args),
        )))
    parts.extend(sorted(f"{a}->{b}" for a, b in g.edges))
    h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return f"spec_{h}"


# ---------------------------------------------------------------------------
# Aggregation — fidelity = module-hierarchy depth
# ---------------------------------------------------------------------------

@dataclass
class AggregatedNode:
    id: str
    label: str
    kind: str          # 'io' | 'mod' | 'op'
    members: list[str]
    shape: Any


@dataclass
class AggregatedGraph:
    level: int
    nodes: list[AggregatedNode]
    edges: list[tuple[str, str]]


def aggregate_at_depth(g: BlockGraph, shapes: dict[str, Any], depth: int) -> AggregatedGraph:
    """Collapse call_module nodes deeper than `depth` into their path[:depth] ancestor.

    Structural ops (call_function, call_method) and I/O always stay individual.
    Two fx nodes that share a path but are siblings (e.g. SPPF's m, m_1, m_2 —
    repeated calls to the same module) are NOT merged: each has a unique
    fx name, so their group key is their own name when their path is at or
    above the target depth.
    """
    name_to_group: dict[str, str] = {}
    groups: dict[tuple, list[GraphNode]] = {}
    group_kind: dict[tuple, str] = {}

    for n in g.nodes:
        if n.op == "call_module" and len(n.path) > depth:
            key = tuple(n.path[:depth])
            kind = "mod"
        else:
            # I/O, ops, or already-shallow modules: each stays its own group
            key = (n.name,)
            if n.op in ("placeholder", "output"):
                kind = "io"
            elif n.op in ("call_function", "call_method"):
                kind = "op"
            else:
                kind = "mod"
        groups.setdefault(key, []).append(n)
        group_kind[key] = kind
        name_to_group[n.name] = "/".join(key)

    nodes: list[AggregatedNode] = []
    for key, members in groups.items():
        gid = "/".join(key)
        kind = group_kind[key]
        if len(members) == 1 and members[0].name == gid:
            n = members[0]
            if n.op == "call_module":
                label = f"{n.name}\n[{n.target_class or n.target}]"
            elif n.op == "call_function":
                fn = n.target.split(".")[-1]
                label = f"fn:{fn}"
            elif n.op == "call_method":
                label = f".{n.target}()"
            elif n.op in ("placeholder", "output"):
                label = n.name
            else:
                label = n.name
        else:
            # Aggregated module group — label is the group path
            label = gid
        last_shape = shapes.get(members[-1].name)
        nodes.append(AggregatedNode(
            id=gid,
            label=label,
            kind=kind,
            members=[m.name for m in members],
            shape=last_shape,
        ))

    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for src, dst in g.edges:
        sg = name_to_group[src]
        dg = name_to_group[dst]
        if sg != dg and (sg, dg) not in seen:
            seen.add((sg, dg))
            edges.append((sg, dg))

    return AggregatedGraph(level=depth, nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Mermaid rendering
# ---------------------------------------------------------------------------

def _shape_str(sh) -> str:
    if not sh:
        return ""
    if isinstance(sh, list) and sh and isinstance(sh[0], int):
        return "×".join(str(s) for s in sh)
    if isinstance(sh, list):
        return " / ".join(_shape_str(x) for x in sh if x)
    return ""


def _sid(s: str) -> str:
    return s.replace("/", "_").replace(".", "_").replace("-", "_")


def render_mermaid(agg: AggregatedGraph, title: str) -> str:
    lines = [f"%% {title}", "flowchart TD"]
    lines.append("  classDef io  fill:#e2e8f0,stroke:#94a3b8,color:#1e293b;")
    lines.append("  classDef mod fill:#dcfce7,stroke:#22c55e,color:#14532d;")
    lines.append("  classDef op  fill:#fed7aa,stroke:#fb923c,color:#7c2d12;")
    lines.append("")

    for n in agg.nodes:
        sh = _shape_str(n.shape)
        lbl = n.label.replace('"', "'").replace("\n", "<br>")
        if sh:
            lbl += f"<br><i>{sh}</i>"
        lines.append(f'  {_sid(n.id)}["{lbl}"]:::{n.kind}')

    lines.append("")
    for a, b in agg.edges:
        lines.append(f"  {_sid(a)} --> {_sid(b)}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Library builder
# ---------------------------------------------------------------------------

def build_library(weights: str, image_path: Path, imgsz: int) -> dict:
    print(f"  loading model {weights}...")
    yolo = load_model(weights)
    blocks = get_blocks(yolo)
    print(f"  capturing per-block input shapes from {image_path}...")
    input_tensor = _preprocess_for_raw_forward(image_path, imgsz)
    in_shapes = capture_block_input_shapes(yolo, input_tensor)

    specs: dict[str, dict] = {}
    instances: list[dict] = []

    for idx, block in enumerate(blocks):
        cls = type(block).__name__
        in_sh = in_shapes.get(idx)
        graph, shapes = derive_block_graph(block, in_sh)
        spec_id = graph_topology_hash(graph)

        # Output shape: dereference the 'output' fx node's input
        out_sh = None
        for n in graph.nodes:
            if n.op == "output":
                ref = n.args[0] if isinstance(n.args, list) and n.args else n.args
                if isinstance(ref, str):
                    out_sh = shapes.get(ref)
                elif isinstance(ref, list) and ref and isinstance(ref[0], str):
                    out_sh = shapes.get(ref[0])
                break

        params = sum(p.numel() for p in block.parameters() if p.requires_grad)

        if spec_id not in specs:
            specs[spec_id] = {
                "spec_id": spec_id,
                "class_name": graph.class_name,
                "signature": graph.signature,
                "derivation_method": graph.derivation_method,
                "derivation_error": graph.derivation_error,
                "forward_source": graph.forward_source,
                # Map of every visible_path prefix to its class. Lets the viewer
                # label aggregated groups correctly without resolving modules.
                "path_classes": build_path_class_map(block),
                "graph": {
                    "nodes": [
                        {
                            "name": n.name,
                            "op": n.op,
                            "target": n.target,
                            "target_class": n.target_class,
                            "path": n.path,
                            "visible_path": n.visible_path,
                            "args": n.args,
                        }
                        for n in graph.nodes
                    ],
                    "edges": [list(e) for e in graph.edges],
                },
                "instances": [],
            }
        specs[spec_id]["instances"].append(idx)

        instances.append({
            "idx": idx,
            "class_name": cls,
            "spec_id": spec_id,
            "input_shape": list(in_sh) if isinstance(in_sh, tuple) else in_sh,
            "output_shape": out_sh,
            "params": params,
            "shapes_by_node": shapes,
        })

        print(
            f"  [{idx:>2}] {cls:<10} -> {spec_id} "
            f"({graph.derivation_method}, nodes={len(graph.nodes)})"
        )

    # Block-to-block connectivity, derived from each block's Ultralytics `.f`
    # attribute (the true data-flow DAG — no hand-coding). `.f` is an int or
    # list of ints; -1 means "immediately preceding block". Sources < 0 after
    # resolution are the model input, not a block-to-block edge.
    edges: list[dict] = []
    for idx, block in enumerate(blocks):
        for src in _resolve_sources(getattr(block, "f", -1), idx):
            if src < 0:
                continue
            edges.append({"src": src, "dst": idx, "is_skip": src != idx - 1})

    # Concat blocks take a list input that fx can't symbolically trace, so
    # their spec graph fails and output_shape comes back None. But a Concat's
    # output is just the channel-wise concatenation of its source blocks'
    # outputs — and the topology above tells us exactly which sources. Fill in
    # input/output shapes from the sources. (`.d` is the concat dim, usually 1.)
    by_idx = {inst["idx"]: inst for inst in instances}
    for inst in instances:
        if inst["class_name"] != "Concat" or inst["output_shape"] is not None:
            continue
        src_shapes = [
            by_idx[e["src"]]["output_shape"] for e in edges if e["dst"] == inst["idx"]
        ]
        if not src_shapes or any(not isinstance(sh, list) for sh in src_shapes):
            continue
        dim = int(getattr(blocks[inst["idx"]], "d", 1))
        out = list(src_shapes[0])
        out[dim] = sum(sh[dim] for sh in src_shapes)
        inst["input_shape"] = [list(sh) for sh in src_shapes]
        inst["output_shape"] = out

    return {
        "model_id": weights.replace(".pt", ""),
        "specs": specs,
        "instances": instances,
        "edges": edges,
    }


def render_block_at_levels(
    library: dict, idx: int, max_level: int = 3
) -> dict[int, str]:
    """For one instance, render Mermaid at L0..L_max_level."""
    inst = next((i for i in library["instances"] if i["idx"] == idx), None)
    if inst is None:
        raise ValueError(f"no instance with idx {idx}")
    spec = library["specs"][inst["spec_id"]]
    if spec["derivation_method"] != "fx":
        return {0: f'%% block {idx}: derivation failed: {spec["derivation_error"]}\n'}

    nodes = [
        GraphNode(
            name=nd["name"],
            op=nd["op"],
            target=nd["target"],
            target_class=nd.get("target_class"),
            path=nd["path"],
            visible_path=nd.get("visible_path", nd["path"]),
            args=nd["args"],
        )
        for nd in spec["graph"]["nodes"]
    ]
    edges = [tuple(e) for e in spec["graph"]["edges"]]
    g = BlockGraph(
        class_name=spec["class_name"],
        signature=spec["signature"],
        nodes=nodes,
        edges=edges,
        forward_source=spec["forward_source"],
        derivation_method="fx",
    )
    shapes = inst["shapes_by_node"]

    out: dict[int, str] = {}

    # L0 — single node containing the whole block
    members = [n.name for n in nodes]
    out[0] = render_mermaid(
        AggregatedGraph(
            level=0,
            nodes=[
                AggregatedNode(
                    id="block",
                    label=f"[{idx}] {spec['class_name']}",
                    kind="mod",
                    members=members,
                    shape=inst["output_shape"],
                )
            ],
            edges=[],
        ),
        title=f"Block {idx} {spec['class_name']} L0",
    )

    for d in range(1, max_level + 1):
        agg = aggregate_at_depth(g, shapes, d)
        out[d] = render_mermaid(
            agg, title=f"Block {idx} {spec['class_name']} L{d}"
        )

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="assets/sammit_lighthouse.jpg", type=Path)
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--demo-block", default=9, type=int)
    p.add_argument("--max-level", default=4, type=int)
    args = p.parse_args()

    out = Path("out/specs")
    out.mkdir(parents=True, exist_ok=True)

    library = build_library(args.weights, args.image, args.imgsz)
    lib_path = out / "library.json"
    lib_path.write_text(json.dumps(library, indent=2, default=str))

    # Also write a JS-wrapped copy so the frontend can read it without fetch()
    # (the page may be opened via file://). Same JSON wrapped in a global.
    js_path = Path("frontend/spec-data.js")
    js_path.write_text(
        "// Auto-generated by yolovex.block_spec — do not edit by hand.\n"
        "window.YV_SPEC = "
        + json.dumps(library, default=str)
        + ";\n"
    )

    print()
    print(f"wrote library to {lib_path}")
    print(f"  unique specs: {len(library['specs'])}")
    print(f"  instances:    {len(library['instances'])}")
    print()
    print(f"{'spec_id':<25} {'class':<10} {'method':<8} {'nodes':<6} instances")
    print("-" * 80)
    for sid, sp in library["specs"].items():
        n_nodes = len(sp["graph"]["nodes"])
        print(
            f"{sid:<25} {sp['class_name']:<10} {sp['derivation_method']:<8} "
            f"{n_nodes:<6} {sp['instances']}"
        )

    print(f"\nrendering demo block {args.demo_block} at L0..L{args.max_level}...")
    levels = render_block_at_levels(library, args.demo_block, args.max_level)
    sample_dir = out / "sample"
    sample_dir.mkdir(exist_ok=True)
    for old in sample_dir.glob(f"{args.demo_block:02d}_L*.mmd"):
        old.unlink()
    for level, mmd in levels.items():
        path = sample_dir / f"{args.demo_block:02d}_L{level}.mmd"
        path.write_text(mmd)
    print(f"  wrote {len(levels)} files to {sample_dir}/")


if __name__ == "__main__":
    main()
