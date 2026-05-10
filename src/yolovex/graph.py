"""SVG architecture graph for yolo26.

Topology is extracted from the model itself (truth from `.f` attrs).
Layout is hand-coded per-block (column + y-rank) to mirror the paper diagram.
The layout is the only thing here that's specific to yolo26's structure —
swap the dicts for a different model variant.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from .topology import GraphEdge, GraphNode

# ----------------------------------------------------------------------------
# yolo26-specific layout. Tweak these dicts and re-render to iterate on layout.
# ----------------------------------------------------------------------------

# Column 0 = backbone, 1 = neck FPN-up, 2 = neck PAN-down, 3 = head
COLUMN: dict[int, int] = {
    **{i: 0 for i in range(0, 11)},   # 0..10 backbone (incl. SPPF & C2PSA)
    **{i: 1 for i in range(11, 17)},  # 11..16 FPN-up
    **{i: 2 for i in range(17, 23)},  # 17..22 PAN-down
    23: 3,                             # Detect head
}

# Y rank within column. Tuned so that skip connections route nearly horizontal:
#   - bb 6 ↔ Concat 12 both sit at y=6
#   - bb 4 ↔ Concat 15 both sit at y=4
#   - FPN-up 13 ↔ PAN-down Concat 18 both sit at y=5
#   - bb 10 (C2PSA) ↔ PAN-down Concat 21 close to y=10
Y_RANK: dict[int, float] = {
    # backbone — linear top-down
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    # neck FPN-up — uniform spacing, climbs from y=8 to y=3
    # (skips: bb 6 → 12 slight diagonal; bb 4 → 15 perfect horizontal at y=4)
    11: 8, 12: 7, 13: 6, 14: 5, 15: 4, 16: 3,
    # neck PAN-down — uniform-ish spacing from 17 (top) descending to 22 (bottom)
    # (skips: FPN 13 → 18 slight diagonal; bb 10 → 21 perfect horizontal at y=10)
    17: 4, 18: 5, 19: 6, 20: 7, 21: 10, 22: 11,
    # head — between its three inputs (P3=3, P4=6, P5=11) ≈ 6.5
    23: 6.5,
}

# Echoes of the paper colours
COLORS: dict[str, str] = {
    "Conv":     "#c8e6c9",
    "C3k2":     "#f8bbd0",
    "C3":       "#f8bbd0",
    "C2f":      "#f8bbd0",
    "Upsample": "#f06292",
    "Concat":   "#d1c4e9",
    "SPPF":     "#fff59d",
    "C2PSA":    "#bbdefb",
    "Detect":   "#66bb6a",
}
DEFAULT_COLOR = "#eeeeee"

# Geometry
NODE_W = 170
NODE_H = 56
ROW_PITCH = 80
TOP_MARGIN = 90
BOTTOM_MARGIN = 60
COL_X: dict[int, int] = {0: 120, 1: 400, 2: 700, 3: 980}

# Group bounding boxes (x ranges per labeled section)
GROUPS = [
    ("Backbone", "#2e7d32", 40,  300,  0,    10),
    ("Neck",     "#1565c0", 340, 880,  3,    11),
    ("Head",     "#c62828", 920, 1170, 6,    7),
]


def _node_center(n: GraphNode) -> tuple[float, float]:
    cx = COL_X[COLUMN[n.idx]] + NODE_W / 2
    cy = TOP_MARGIN + Y_RANK[n.idx] * ROW_PITCH + NODE_H / 2
    return cx, cy


def _node_svg(n: GraphNode) -> str:
    cx, cy = _node_center(n)
    x = cx - NODE_W / 2
    y = cy - NODE_H / 2
    color = COLORS.get(n.type_name, DEFAULT_COLOR)
    return f'''<g class="node" data-idx="{n.idx}">
  <rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="6"
        fill="{color}" stroke="#222" stroke-width="1.2"/>
  <text x="{cx:.1f}" y="{y + 22:.1f}" text-anchor="middle"
        font-family="-apple-system, BlinkMacSystemFont, sans-serif"
        font-size="13" font-weight="600">[{n.idx}] {xml_escape(n.type_name)}</text>
  <text x="{cx:.1f}" y="{y + 42:.1f}" text-anchor="middle"
        font-family="ui-monospace, SFMono-Regular, monospace"
        font-size="10" fill="#333">{xml_escape(n.output_shape)}</text>
</g>'''


def _edge_svg(e: GraphEdge, by_idx: dict[int, GraphNode]) -> str:
    src, dst = by_idx[e.src], by_idx[e.dst]
    sx, sy = _node_center(src)
    dx, dy = _node_center(dst)
    src_col, dst_col = COLUMN[src.idx], COLUMN[dst.idx]

    if src_col == dst_col:
        # within-column: bottom of upper → top of lower (or reverse)
        if dy > sy:
            x0, y0, x1, y1 = sx, sy + NODE_H / 2, dx, dy - NODE_H / 2
        else:
            x0, y0, x1, y1 = sx, sy - NODE_H / 2, dx, dy + NODE_H / 2
        path = f"M {x0:.1f} {y0:.1f} L {x1:.1f} {y1:.1f}"
    else:
        # cross-column: right side of source → left side of dest, smooth Bezier
        x0, y0 = sx + NODE_W / 2, sy
        x1, y1 = dx - NODE_W / 2, dy
        mid = (x0 + x1) / 2
        path = f"M {x0:.1f} {y0:.1f} C {mid:.1f} {y0:.1f}, {mid:.1f} {y1:.1f}, {x1:.1f} {y1:.1f}"

    if e.is_skip:
        stroke, width, marker = "#fb923c", 2.0, "arrow-skip"
    else:
        stroke, width, marker = "#555", 1.5, "arrow-fwd"
    return (
        f'<path d="{path}" stroke="{stroke}" stroke-width="{width}" '
        f'fill="none" marker-end="url(#{marker})"/>'
    )


def _group_svg(label: str, color: str, x0: int, x1: int, y_min: float, y_max: float) -> str:
    y0 = TOP_MARGIN + y_min * ROW_PITCH - 36
    y1 = TOP_MARGIN + y_max * ROW_PITCH + NODE_H + 24
    return f'''<g class="group">
  <rect x="{x0}" y="{y0:.1f}" width="{x1 - x0}" height="{y1 - y0:.1f}"
        fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="6,4" rx="10"/>
  <text x="{x0 + 12}" y="{y0 - 8:.1f}"
        font-family="-apple-system, sans-serif" font-size="15" font-weight="700"
        fill="{color}">{label}</text>
</g>'''


def render_svg(nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
    by_idx = {n.idx: n for n in nodes}

    canvas_w = 1200
    max_y = max(Y_RANK.values())
    canvas_h = int(TOP_MARGIN + (max_y + 1) * ROW_PITCH + BOTTOM_MARGIN)

    edges_svg = "\n  ".join(_edge_svg(e, by_idx) for e in edges)
    nodes_svg = "\n  ".join(_node_svg(n) for n in nodes)
    groups_svg = "\n  ".join(_group_svg(*g) for g in GROUPS)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}"
     width="{canvas_w}" height="{canvas_h}">
  <defs>
    <marker id="arrow-fwd" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>
    </marker>
    <marker id="arrow-skip" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#fb923c"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="white"/>

  {groups_svg}

  <g class="edges">
  {edges_svg}
  </g>

  <g class="nodes">
  {nodes_svg}
  </g>
</svg>
'''


def write_svg(nodes: list[GraphNode], edges: list[GraphEdge], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_svg(nodes, edges))
    return out_path
