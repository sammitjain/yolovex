# yolovex frontend

Single-page React app (UMD + in-browser Babel — no build step). Reads real
activations + predictions from a generated `data.js` file. The frontend ships
**two pages** that share most of the supporting JSX modules:

| Page | What's a node | Data file |
|---|---|---|
| `yolovex.html`     | One per top-level block (24 nodes) — **Level 1** | `data.js`     |
| `yolovex-l2.html`  | Sub-modules inside each block (Conv2d / BN / SiLU; cv1 / m.0 / concat / cv2; etc.) — **Level 2** | `data-l2.js` |

## Run

```bash
# 1. Generate the data files
uv run yolovex build-assets        # → frontend/data.js     (L1, ~580 KB)
uv run yolovex build-assets-l2     # → frontend/data-l2.js  (L2, ~2 MB; runs build-assets first internally)

# 2. Serve (browsers block file:// XHR for the babel JSX scripts)
cd frontend
python -m http.server 8000

# 3. Open
open http://localhost:8000/yolovex.html      # block-level view
open http://localhost:8000/yolovex-l2.html   # sub-module view
```

## Files

### Shared

- `internals.jsx` — "Inside the block" SVG diagrams for all 7 block types (used by L1 panel)
- `heatmap.jsx`   — procedural viridis textures for fallback rendering
- `assets/`       — `sammit_lighthouse.jpg` template

### Level 1 (block-level)

- `yolovex.html`   — page shell + all CSS, loads L1 modules
- `data.js`        — `window.YV_DATA` (24 blocks)
- `arch.jsx`       — static architecture metadata + edges
- `layout.jsx`     — pixel positions + bezier paths
- `graph.jsx`      — interactive SVG graph (pan/zoom/hover/click/play-flow)
- `panel.jsx`      — detail panel (FeaturePanel for blocks, DetectPanel for the head)
- `app.jsx`        — main composition + image picker

### Level 2 (sub-module)

- `yolovex-l2.html` — page shell, loads L2 modules
- `data-l2.js`      — `window.YV_DATA` for L2 (89 entries — 24 parents + 65 sub-nodes; lookup by string ID like `"4.cv1"`)
- `arch-l2.jsx`     — sub-node templates per parent type, `getArchL2()` builds containers, `buildIntraEdges()` includes type-specific fan-in edges for SPPF / C3k2
- `layout-l2.jsx`   — slot-based row layout (`SUBNODE_LAYOUTS_L2`) with left / right / center / wide cells; container sizes computed per parent type; bus-routed fan-in arrows for SPPF
- `graph-l2.jsx`    — SVG renderer with notched-tail markers (`markerUnits="userSpaceOnUse"`); SubNode component shows type label + tensor shape
- `app-l2.jsx`      — forked from `app.jsx` because L2's selectedId is a string (`"4.cv1"`) and the panels handle different cases (DetectContainerCard with confidence slider, DetectHeadCard for per-scale, FeaturePanelL2 for everything else, FlowOverlay tile)

## L2 architecture notes

### Why a separate data file

`build_assets.py` (L1) goes through Ultralytics' high-level predict path, which
calls `model.fuse()` internally and folds `BatchNorm2d` into the preceding
`Conv2d` weights — at which point BN's forward hook never fires. The L2
builder (`build_assets_l2.py`) does a **second forward pass** by calling
`yolo.model(tensor)` directly on a manually `LetterBox`-preprocessed tensor.
That bypasses the predictor's setup and leaves BN as a separate module so its
hook fires and we get real BN activations. SiLU is shared across all Conv
blocks in yolo26's source, so we use the parent block's L1 output as the act
sub-node's output (mathematically identical — verified to 0 max diff).

### Sub-node ID scheme

Hierarchical strings: `<parent_idx>.<path>`. Examples:

```
"0.conv"   "0.bn"   "0.act"
"4.cv1"    "4.m.0"  "4.concat"   "4.cv2"
"9.cv1"    "9.m.0"  "9.m.1"      "9.m.2"   "9.concat"   "9.cv2"
"23.head_p3"   "23.head_p4"   "23.head_p5"
```

The numeric-string parent IDs (`"0"`, `"4"`, `"23"`) live in the same `blocks`
map alongside the sub-nodes so a single `panel.jsx`-style lookup works for
both granularities.

### Layout rules per block type

`SUBNODE_LAYOUTS_L2` has two flavours:

- **Stack mode** (`{ stack: [...] }`) — simple top-to-bottom column. Every
  sub-node sits at `innerLeft`, width = `TILE_W`. Used by `Conv`, `C2PSA`,
  `Detect`.
- **Staircase mode** (`{ taps, merge, postMerge }`) — used by any block
  where a sub-merge takes N sibling inputs. Each tap `i` shifts right by
  `i × STAIR_FRAC × TILE_W` (default `STAIR_FRAC = 0.65`); the merge tile
  spans the full inner width (`TILE_W × (1 + STAIR_FRAC × (N − 1))`); each
  postMerge tile is `TILE_W` wide left-aligned at `innerLeft` so it lines
  up with the next block's first sub. C3k2 (2 taps) and SPPF (4 taps) use
  this mode.

Layout constants (from `layout-l2.jsx`, also mirrored in
`frontend/sppf-test.html`):

```
TILE_W       = 120   (uniform sub-node width across all blocks)
TILE_H       = 46
STAIR_FRAC   = 0.65
SUB_ROW_GAP  = 78    (top-to-top distance between rows)
SEQ_BEND     = 22    (lead length for the right-angle tap→tap bezier)
COL_GAP_L2   = 520
ATOMIC_W     = TILE_W + 2 × pad = 152
```

### Edge routing

| Kind | Geometry |
|---|---|
| `intra` between two staircase **tap** rows | `rightAngleBezier(rightPort(a), top/botPort(b))` — out from a's right-mid horizontally, then turn vertically into b. Direction picked from y so it works in FPN-up reverse. |
| `intra` between **stack** rows or **merge → post** | Vertical bezier with horizontal-tangent ends. Degenerates to a straight line when source and target share x. |
| `fanin` (tap → merge) | Pure-vertical straight `M ... L ...` line at the tap's center x. **No curve.** The staircase guarantees each tap is at a unique x, so no detour or bus routing is needed. |
| `forward` (same-column inter-container) | Vertical bezier from prev's last sub bottom-mid (or top-mid in FPN-up) to next's first sub top-mid (or bottom-mid). With every tile left-aligned, this collapses to a clean straight vertical line. |
| `cross` (different-column inter-container) | Horizontal bezier `rightPort(a) → leftPort(b)`. |
| `skip` | Long-flat-tail bezier landing on the target container's **left-edge mid**. The c2 control point sits far to the left of p2 at exactly `p2.y` so the curve enters the arrow head along a horizontal tangent. |

Arrow heads are rendered with `markerUnits="userSpaceOnUse"` so they're a
fixed pixel size regardless of stroke width, with a notched-tail triangle
(`M 0 0 L 12 6 L 0 12 L 3 6 z`) — the notch hides any tiny tangent mismatch
and gives the classic draw.io look.

### SPPF / C3k2 layout test canvas

`frontend/sppf-test.html` is a self-contained page that renders ONLY the
SPPF block (container + 6 sub-nodes + edges). All layout constants and
bezier helpers are inlined at the top of the script — no dependency on
`data.js`, `arch-l2.jsx`, or the rest of the L2 app. Open it directly:

```
http://localhost:8000/sppf-test.html
```

Use it whenever the inside-the-block geometry needs another tuning pass.
Once happy, port the constants back into `layout-l2.jsx`.

### Detect card

Click the parent Detect container (idx 23) to see:

1. The input image with classic bbox + class/conf labels overlaid (HTML divs,
   not SVG strokes — gives crisp pixel-width borders and proper text rendering).
2. A **confidence-threshold slider** (default 0.25, range 0.005–1.000) showing
   "N survivors · M runners-up · K total candidates". Sliding lower fades in
   the runners-up boxes (dashed). The L2 builder runs a separate predict pass
   at conf=0.001 to capture all ~50 candidates in `blocks["23"].candidate_boxes`.
3. Per-scale story (P3 / P4 / P5 — top class peak per branch).
4. Compact structural metadata (`nc`, `reg_max`, strides, `no`).

Click an individual `head_pX` sub-node to see that scale's per-class heatmaps
(top 5 classes by peak score) plus the global final-detections list.

### Flow overlay

A fixed tile in the top-right of the canvas (~280 px wide, aspect-ratio 4:5)
that always shows ONE picture at a constant size:

- **Idle / hover / select**: the active layer's `mean_thumbnail`.
- **Active = Detect container**: the input image with bbox overlays.
- **Idle and nothing active**: the original input image.

During play-flow the tile updates at the configured speed (slow / medium /
fast — 700 / 250 / 60 ms per tick). The flow walks through every sub-node in
dataflow order (atomic blocks emit their own frame; non-atomic parents are
skipped because their data is already shown by the last sub-node), and ends
on the Detect container so the final frame is the annotated image.

## Real vs procedural mode

- **Real:** when `imageUrl === window.YV_DATA.meta.image` (the bundled
  lighthouse), every thumbnail is a base64 JPEG of the actual activation.
- **Procedural:** uploaded images fall back to procedural viridis heatmaps
  (L1 only). L2 doesn't yet render procedural thumbnails for unbundled images.

To wire live activations for arbitrary uploads, add a small Python backend
(FastAPI) that runs the model on demand and serves the JPEGs. That's a v2
conversation; the static-data path is fine for the wireframe.

## Regenerating data files

```bash
uv run yolovex build-assets                                  # default lighthouse, L1 only
uv run yolovex build-assets-l2                               # L1 + L2 (calls build-assets internally)
uv run yolovex build-assets --image path/to/other.jpg        # L1 with a different image
uv run yolovex build-assets-l2 --image path/to/other.jpg     # both
```

Sizes (lighthouse): `data.js` ≈ 580 KB, `data-l2.js` ≈ 2 MB. JPEGs are
rendered at native tensor resolution capped at 80 px (channels) / 60 px
(class heatmaps), JPEG quality 85.

## Known follow-ups

- **L0** (role-only) and **L3** (Bottleneck / PSABlock expanded) — same
  architecture as L2; defer until we've stabilized L2.
- **Fidelity scrubber** on a single canvas — design after L0/L3 exist.
- **Per-page spacing controls** — vertical / horizontal gaps as user-tunable
  settings, since the diagram is dense.
- **Port the L2 arrow markers back to L1** so both pages share a visual
  language. Same goes for thicker bbox rendering on L1's predict overlay.
- Lightbulb "insight" tags per block (P3-empty / P4-tie / P5-person teaching
  moments).
- Toggle between mean and max-activated channel for the I/O thumbnails.
- Trace marker coordinate scaling currently ignores letterboxing — fix when
  revisiting trace UX.
- Live uploads with real activations → small FastAPI backend (v2).
