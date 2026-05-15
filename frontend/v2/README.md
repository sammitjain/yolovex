# yolovex v2 — implementation notes

A fresh rebuild of the interactive YOLO26 explainer. Currently at **L1**: one
node per block (24 blocks), positioned, connected, and grouped into role
containers. Deeper levels and explanation/activation overlays are future work.

## Source of truth

The architecture is **derived**, not hand-coded. The spec-viewer pipeline
(`src/yolovex/block_spec.py` → `frontend/spec-data.js`) emits
`window.YV_SPEC`:

- `specs` — deduplicated per-block internal fx graphs (used by the spec viewer;
  v2 will use these for the future in-place expansion).
- `instances` — the 24 blocks: `idx`, `class_name`, `input_shape`,
  `output_shape`, `params`, `shapes_by_node`.
- `edges` — block-to-block connectivity `{src, dst, is_skip}`, derived by
  walking each block's Ultralytics `.f` attribute (reuses `_resolve_sources`
  from `topology.py`). **No hand-coded wiring.**

Concat blocks can't be fx-traced (list inputs), so the pipeline back-fills
their `input_shape` / `output_shape` from their source blocks' shapes via the
topology (channel-wise sum). Verified against the original L1's hand-coded
values.

Regenerate after a model/pipeline change: `uv run python -m yolovex.block_spec`

## Files (`frontend/v2/`)

| File | Role |
|---|---|
| `yolovexv2.html` | Entry point. Loads React/Babel + `../spec-data.js` + the 5 jsx files. |
| `arch-v2.jsx` | Data model. Joins `YV_SPEC.instances` + `edges` with a small **presentation** config (`PRESENTATION`: per-block `col`/`vpos`/`role`). Roles: Backbone 0–8, Neck **9–22**, Head 23. Type/role colors. |
| `layout-v2.jsx` | Pixel positions, role-container shapes, edge paths. Columns are placed as flow-order walks so expanded blocks slide neighbours aside. **All tunable spacing lives at the top.** |
| `expand-v2.jsx` | In-place expansion. Self-contained port of the spec-viewer graph machinery + `buildExpansion(idx, {flip})` — turns one L1 block into a laid-out internal component sub-graph (depth-1 aggregation for now). |
| `graph-v2.jsx` | SVG render + pan/zoom/hover. Clicking a block toggles expansion; renders expanded regions and their internal sub-graph. |
| `app-v2.jsx` | App shell — header + `GraphV2`, holds `hoveredId`/`selectedId`/`expandedCount`. |

## Tunable spacing — top of `layout-v2.jsx`

```
ROW_GAP        vertical gap between blocks in a column
COL_GAP        horizontal gap between the two neck columns (FPN-up / PAN-down)
CONTAINER_GAP  uniform horizontal gap between Backbone | Neck | Head containers
NECK_Y_OFFSET  how far down the whole neck region sits (breathing room below backbone)
DETECT_GAP     vertical gap between the P3 / P4 / P5 detect boxes
H_ENTRY/H_EXIT flat-tail leads for horizontal edges (cross / skip / detect)
V_ENTRY/V_EXIT flat-tail leads for vertical edges (forward / same-column)
```

`COL_X` is derived from these so the three container gaps stay uniform while
`COL_GAP` independently controls the neck's internal column spacing. The neck
body's left edge hugs its first column (`xDivide = body.x - CONTAINER_PAD`),
so widening `CONTAINER_GAP` no longer opens dead space inside the neck.

## Layout specifics

- **Positions** reproduce the original `yolovex.html` column scheme (ported
  from `frontend/layout.jsx`), plus `NECK_Y_OFFSET`.
- **Mirrored-L Neck container** — SPPF [9] / C2PSA [10] sit in the backbone
  *column* but belong to the Neck *role*, so the Neck container is an
  L-polygon: the cols-1&2 body, with a down-left "foot" enclosing 9 & 10. The
  foot is intentionally wide (it spans out to the body's left edge — that
  space is where 9/10's skip + cross edges route). Backbone (0–8) and Head
  (23) are plain rounded rects. See `computeContainers` + `roundedPolyPath`.
- **Detect head** — three separate normal-sized boxes (P3/P4/P5) stacked with
  `DETECT_GAP`, not one stretched node. `node.detect[]` carries each box's
  `relY`; `detectPort` returns the matching box's left port.
- **Edges** — flat-tail beziers ported from `layout-l2.jsx`
  (`flatBezier` / `flatBezierVertical` / `fitTails`): a flat lead out of the
  source, a short bend, a flat lead into the target. `buildEdgeMeta` derives
  the rendering `kind` (forward / cross / skip / detect) from the topology.

## Previewing

`.claude/launch.json` defines a `node`-based static server (`yolovex-v2`,
port 8765) that redirects `/` → `/v2/yolovexv2.html` so relative paths
resolve. If opening directly, serve the `frontend/` dir and open
`/v2/yolovexv2.html` — `arch-v2.jsx` etc. are relative to `/v2/`, and
`../spec-data.js` must resolve. (A stale cached `spec-data.js` without the
`edges` key will make all connections disappear — hard-refresh.)

## Click-to-expand (built)

Clicking an L1 block expands it in place to its internal component blocks
(depth-1 aggregation — the "first decomposition", e.g. Conv → conv/bn/act).
Neighbours **translate** (never resize) to make room: each column is a
flow-order walk, so blocks after an expanded one slide along the flow
direction, and columns to the right shift over by the region's extra width.
Neck FPN-up blocks (col 1) `flip` their internal flow bottom-to-top. The
expanded block's I/O placeholders are dropped — the previous block connects to
the first internal layer, the last internal layer connects onward. Multiple
blocks can be expanded at once; the `⊟` control collapses all.

## Next steps (not yet built)

1. **Recursive / deeper expansion** — a per-region fidelity control to peel
   further levels (the spec-viewer aggregation already supports it).
2. **Explanation / activation overlays** — the pedagogical layer
   (feature-map thumbnails, flow animation, detection viz) — see the original
   `yolovex-l2.html` / `app-l2.jsx` for the patterns to draw from.
