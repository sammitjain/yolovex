"""Extract the model's data-flow topology from `.f` attributes.

Every Ultralytics block carries a `.f` attribute giving the indices of its
input blocks (or `-1` for "the immediately preceding block"). Walking that
gives us the true DAG without any hand-coding.
"""

from __future__ import annotations

from dataclasses import dataclass

from .layers import shape_of


@dataclass
class GraphNode:
    idx: int
    type_name: str
    sources: list[int]   # block indices feeding this node
    output_shape: str    # human-readable shape string


@dataclass
class GraphEdge:
    src: int
    dst: int
    is_skip: bool        # True iff the source is NOT the immediate predecessor


def _resolve_sources(f, idx: int) -> list[int]:
    """`.f` is either an int (often -1) or a list of ints. -1 means `idx - 1`."""
    raw = f if isinstance(f, list) else [f]
    return [idx - 1 if x == -1 else int(x) for x in raw]


def extract_topology(blocks, activations) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for i, b in enumerate(blocks):
        sources = _resolve_sources(b.f, i)
        nodes.append(
            GraphNode(
                idx=i,
                type_name=type(b).__name__,
                sources=sources,
                output_shape=shape_of(activations.get(i)),
            )
        )
        for src in sources:
            if src < 0:
                continue  # source -1 == model input; not a block-to-block edge
            edges.append(GraphEdge(src=src, dst=i, is_skip=(src != i - 1)))
    return nodes, edges


def topology_summary(nodes: list[GraphNode]) -> str:
    """Print-friendly table for eyeballing against the paper diagram."""
    lines = [f"{'idx':>3}  {'type':<10}  {'sources':<14}  output shape"]
    lines.append("-" * 60)
    for n in nodes:
        src_str = f"[{', '.join(map(str, n.sources))}]"
        lines.append(f"{n.idx:>3}  {n.type_name:<10}  {src_str:<14}  {n.output_shape}")
    return "\n".join(lines)
