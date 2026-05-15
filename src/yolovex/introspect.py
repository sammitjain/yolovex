"""Per-block model introspection.

Walks every top-level YOLO block, attempts torch.fx symbolic tracing scoped
to that block, runs ShapeProp to annotate tensor shapes, and dumps:

  out/specs/<idx>_<class>.txt    human-readable graph table + signature
  out/specs/<idx>_<class>.json   machine-readable graph (for BLOCK_SPEC work)
  out/specs/<idx>_<class>.mmd    mermaid flowchart for visual inspection

When symbolic_trace fails on a block, the .txt / .json files still get
written with the error message and the forward() source so we know what we
were up against. Run with:

    uv run python -m yolovex.introspect

Flags:
    --image PATH      input image (default: assets/sammit_lighthouse.jpg)
    --imgsz N         square input size (default: 640)
    --weights NAME    model weights (default: yolo26n.pt)
    --only IDX,IDX    introspect only these block indices

This is exploratory tooling — outputs go under out/specs/ which is gitignored.
"""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from .build_assets_l2 import _preprocess_for_raw_forward
from .model import DEFAULT_WEIGHTS, get_blocks, load_model


# ---------------------------------------------------------------------------
# Structural signature — captures actual sub-module shape, not just classname
# ---------------------------------------------------------------------------

def structural_signature(module: torch.nn.Module, depth: int = 1) -> str:
    """Recursive class-name signature down `depth` levels.

    `C3k2(cv1=Conv,m=ModuleList[Bottleneck],cv2=Conv)` distinguishes block 4
    from block 6 even though both are class C3k2.
    """
    parts = []
    for name, child in module.named_children():
        cls = type(child).__name__
        if cls in ("ModuleList", "Sequential") and len(list(child.children())):
            inner = type(next(iter(child.children()))).__name__
            n = len(list(child.children()))
            parts.append(f"{name}={cls}[{inner}x{n}]")
        elif depth > 0 and len(list(child.children())):
            inner_sig = structural_signature(child, depth - 1)
            parts.append(f"{name}={cls}{inner_sig}")
        else:
            parts.append(f"{name}={cls}")
    return f"({','.join(parts)})" if parts else "()"


# ---------------------------------------------------------------------------
# Per-block input-shape capture (one forward pass, pre-hooks on every block)
# ---------------------------------------------------------------------------

def capture_block_input_shapes(yolo, input_tensor) -> dict[int, tuple[int, ...]]:
    """Run a dummy forward and record the shape entering each top-level block."""
    shapes: dict[int, tuple[int, ...]] = {}
    blocks = get_blocks(yolo)
    handles = []

    def make_hook(idx):
        def hook(_mod, inputs):
            x = inputs[0] if inputs else None
            if torch.is_tensor(x):
                shapes[idx] = tuple(x.shape)
            elif isinstance(x, (list, tuple)) and x and torch.is_tensor(x[0]):
                # Some blocks (Concat, Detect) take a list of tensors
                shapes[idx] = tuple(("list", len(x), tuple(x[0].shape)))
        return hook

    for idx, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            yolo.model(input_tensor)
    finally:
        for h in handles:
            h.remove()

    return shapes


# ---------------------------------------------------------------------------
# Per-block FX trace
# ---------------------------------------------------------------------------

@dataclass
class BlockReport:
    idx: int
    class_name: str
    signature: str
    input_shape: Any
    forward_source: str
    fx_ok: bool
    fx_error: str | None = None
    nodes: list[dict] = field(default_factory=list)


def _shape_of(node) -> Any:
    meta = getattr(node, "meta", {}) or {}
    tm = meta.get("tensor_meta")
    if tm is None:
        return None
    # TensorMetadata is a namedtuple — check .shape BEFORE the tuple/list branch
    # (otherwise the namedtuple's fields get iterated as if they were tensors).
    if hasattr(tm, "shape"):
        return list(tm.shape)
    if isinstance(tm, (list, tuple)):
        return [list(t.shape) if hasattr(t, "shape") else None for t in tm]
    return None


def _arg_repr(a) -> Any:
    """Stringify fx args: Nodes become their .name; everything else is JSON-able."""
    if isinstance(a, torch.fx.Node):
        return a.name
    if isinstance(a, (list, tuple)):
        return [_arg_repr(x) for x in a]
    if isinstance(a, dict):
        return {k: _arg_repr(v) for k, v in a.items()}
    if isinstance(a, (int, float, str, bool)) or a is None:
        return a
    return repr(a)


def introspect_block(idx: int, block: torch.nn.Module, input_shape: Any) -> BlockReport:
    cls = type(block).__name__
    sig = structural_signature(block, depth=1)
    src = inspect.getsource(block.__class__.forward)

    rep = BlockReport(
        idx=idx,
        class_name=cls,
        signature=sig,
        input_shape=input_shape,
        forward_source=src,
        fx_ok=False,
    )

    # We can only fx-trace tensor-in/tensor-out modules with a known input shape.
    if not isinstance(input_shape, tuple) or input_shape and input_shape[0] == "list":
        rep.fx_error = f"non-tensor input ({input_shape}); skipped fx trace"
        return rep

    try:
        gm = torch.fx.symbolic_trace(block)
        try:
            example = torch.zeros(*input_shape)
            ShapeProp(gm).propagate(example)
        except Exception as e:
            rep.fx_error = f"trace OK, ShapeProp failed: {e}"
        rep.fx_ok = True
        for n in gm.graph.nodes:
            rep.nodes.append({
                "name": n.name,
                "op": n.op,
                "target": n.target if isinstance(n.target, str) else getattr(n.target, "__qualname__", None) or getattr(n.target, "__name__", None) or repr(n.target),
                "args": _arg_repr(n.args),
                "kwargs": _arg_repr(dict(n.kwargs)) if n.kwargs else {},
                "shape": _shape_of(n),
            })
    except Exception as e:
        rep.fx_error = f"symbolic_trace failed: {e.__class__.__name__}: {e}"

    return rep


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_text(rep: BlockReport, path: Path) -> None:
    lines = []
    lines.append(f"# Block [{rep.idx}] {rep.class_name}")
    lines.append("")
    lines.append(f"signature: {rep.class_name}{rep.signature}")
    lines.append(f"input_shape: {rep.input_shape}")
    lines.append("")
    lines.append("---- forward() source ----")
    lines.append(rep.forward_source.rstrip())
    lines.append("")
    lines.append("---- fx graph ----")
    if not rep.fx_ok:
        lines.append(f"FX UNAVAILABLE: {rep.fx_error}")
    else:
        if rep.fx_error:
            lines.append(f"(note) {rep.fx_error}")
        # Tabular print
        header = f"{'name':<22} {'op':<14} {'target':<28} {'shape':<22} args"
        lines.append(header)
        lines.append("-" * len(header))
        for n in rep.nodes:
            shape = str(n["shape"]) if n["shape"] is not None else "—"
            target = n["target"][:28]
            lines.append(f"{n['name']:<22} {n['op']:<14} {target:<28} {shape:<22} {n['args']}")
    path.write_text("\n".join(lines) + "\n")


def write_json(rep: BlockReport, path: Path) -> None:
    path.write_text(json.dumps({
        "idx": rep.idx,
        "class_name": rep.class_name,
        "signature": rep.signature,
        "input_shape": list(rep.input_shape) if isinstance(rep.input_shape, tuple) else rep.input_shape,
        "fx_ok": rep.fx_ok,
        "fx_error": rep.fx_error,
        "forward_source": rep.forward_source,
        "nodes": rep.nodes,
    }, indent=2, default=str))


def write_mermaid(rep: BlockReport, path: Path) -> None:
    """Render FX graph as a Mermaid flowchart.

    Nodes are colored by op:
      - placeholder/output: gray
      - call_module:        green   (sub-module, has weights)
      - call_function:      orange  (torch op like cat, chunk, add)
      - call_method:        blue    (tensor method like .chunk())
      - get_attr:           purple
    """
    if not rep.fx_ok or not rep.nodes:
        path.write_text(f"%% fx unavailable for block {rep.idx}: {rep.fx_error}\n")
        return

    op_classes = {
        "placeholder":   "ph",
        "output":        "ph",
        "call_module":   "mod",
        "call_function": "fn",
        "call_method":   "mth",
        "get_attr":      "attr",
    }
    node_class_lines = {
        "ph":   "classDef ph   fill:#e2e8f0,stroke:#94a3b8,color:#1e293b;",
        "mod":  "classDef mod  fill:#dcfce7,stroke:#22c55e,color:#14532d;",
        "fn":   "classDef fn   fill:#fed7aa,stroke:#fb923c,color:#7c2d12;",
        "mth":  "classDef mth  fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;",
        "attr": "classDef attr fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;",
    }

    lines = ["flowchart TD"]
    for cd in node_class_lines.values():
        lines.append(f"  {cd}")
    lines.append("")

    nodes_by_name = {n["name"]: n for n in rep.nodes}

    for n in rep.nodes:
        cls = op_classes.get(n["op"], "fn")
        shape = n["shape"]
        if isinstance(shape, list) and shape and isinstance(shape[0], int):
            shape_str = "×".join(str(s) for s in shape)
        elif isinstance(shape, list):
            shape_str = "/".join("×".join(str(s) for s in (sh or [])) for sh in shape)
        else:
            shape_str = ""
        target = n["target"]
        # Build label
        if n["op"] == "call_module":
            label = f"{n['name']}<br><i>{target}</i>"
        elif n["op"] == "call_function":
            short = target.split(".")[-1].rstrip(">")
            label = f"{n['name']}<br><i>fn:{short}</i>"
        elif n["op"] == "call_method":
            label = f"{n['name']}<br><i>.{target}()</i>"
        else:
            label = n["name"]
        if shape_str:
            label += f"<br>{shape_str}"
        # Mermaid IDs: alphanumeric only
        nid = n["name"].replace(".", "_")
        lines.append(f"  {nid}[\"{label}\"]:::{cls}")

    lines.append("")
    # Edges: walk every node's args; any arg that names another node → edge
    def collect_refs(arg, refs):
        if isinstance(arg, str) and arg in nodes_by_name:
            refs.add(arg)
        elif isinstance(arg, list):
            for x in arg:
                collect_refs(x, refs)
        elif isinstance(arg, dict):
            for v in arg.values():
                collect_refs(v, refs)

    for n in rep.nodes:
        refs: set[str] = set()
        collect_refs(n["args"], refs)
        collect_refs(n["kwargs"], refs)
        for src in refs:
            src_id = src.replace(".", "_")
            dst_id = n["name"].replace(".", "_")
            lines.append(f"  {src_id} --> {dst_id}")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(image_path: Path, weights: str, imgsz: int, only: list[int] | None) -> None:
    out_dir = Path("out/specs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading model ({weights})...")
    yolo = load_model(weights)
    blocks = get_blocks(yolo)
    print(f"  model has {len(blocks)} top-level blocks")

    print(f"  capturing per-block input shapes from {image_path}...")
    input_tensor = _preprocess_for_raw_forward(image_path, imgsz)
    in_shapes = capture_block_input_shapes(yolo, input_tensor)

    summary = []
    for idx, block in enumerate(blocks):
        if only is not None and idx not in only:
            continue
        cls = type(block).__name__
        in_shape = in_shapes.get(idx, "unknown")
        print(f"  [{idx:>2}] {cls:<10} in={in_shape} ", end="", flush=True)
        try:
            rep = introspect_block(idx, block, in_shape)
        except Exception as e:
            traceback.print_exc()
            print(f"hard fail: {e}")
            continue
        base = out_dir / f"{idx:02d}_{cls}"
        write_text(rep, base.with_suffix(".txt"))
        write_json(rep, base.with_suffix(".json"))
        write_mermaid(rep, base.with_suffix(".mmd"))
        status = "ok" if rep.fx_ok else f"fx-fail"
        node_count = len(rep.nodes) if rep.fx_ok else 0
        print(f"-> {status}  nodes={node_count}  sig={cls}{rep.signature}")
        summary.append((idx, cls, rep.fx_ok, node_count, rep.signature))

    print()
    print(f"wrote {len(summary)} block reports to {out_dir}/")
    print()
    print(f"{'idx':<4} {'class':<10} {'fx':<6} {'nodes':<6} signature")
    print("-" * 80)
    for idx, cls, ok, nc, sig in summary:
        print(f"{idx:<4} {cls:<10} {'OK' if ok else 'FAIL':<6} {nc:<6} {cls}{sig}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="assets/sammit_lighthouse.jpg", type=Path)
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--only", default=None, help="comma-separated block indices, e.g. 4,6,9")
    args = p.parse_args()
    only = None
    if args.only:
        only = [int(x) for x in args.only.split(",")]
    run(args.image, args.weights, args.imgsz, only)


if __name__ == "__main__":
    main()
