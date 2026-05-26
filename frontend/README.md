# yolovex frontend — implementation notes

The interactive explorer. Static HTML + React (via Babel-standalone in the
browser, so no build step). It reads two generated payloads:

- `spec-data.js` — `window.YV_SPEC`: deduplicated block specs, instances,
  and the data-flow edges. Produced by `yolovex.block_spec`.
- `activations.js` — `window.YV_ACT`: per-fx-node activations + Detect
  payload (boxes, per-class score maps) for one image. Produced by
  `yolovex build-assets` (or rewritten on each upload by `yolovex serve`).

## Files

| File | Role |
| --- | --- |
| `index.html` | Entry point. Loads React/Babel + the two payloads + the five jsx files. |
| `arch.jsx` | Data model. Joins `YV_SPEC.instances` + `edges` with a small **presentation** config (`PRESENTATION`: per-block `col`/`vpos`/`role`). Roles: Backbone 0–8, Neck 9–22, Head 23. Type/role palettes for light + dark. |
| `layout.jsx` | Pixel positions, role-container shapes, edge paths. Columns are flow-order walks so expanded blocks slide neighbours aside. All tunable spacing lives at the top of this file. |
| `expand.jsx` | In-place expansion. Self-contained graph machinery + `buildExpansion(idx, {flip})` — turns one L1 block into a laid-out internal component sub-graph (recursive via the `opts.expansions` set). |
| `graph.jsx` | SVG render + pan/zoom/hover. Clicking a block toggles expansion; renders expanded regions and their internal sub-graph. |
| `app.jsx` | App shell — header, flow play, Settings drawer, DetailPanel, DetectPanel, the 📷 upload flow (server mode), and the `BuildProgressOverlay`. |
| `spec-data.js` | Generated. Architecture spec — committed alongside source. |
| `activations.js` | Generated. Per-image activations — regenerated on every `build-assets` / upload. |
| `design-spec.html` | Static design doc; not loaded at runtime. |

## Window namespace

All app-internal modules attach to `window.YV` (e.g. `window.YV.buildArch`,
`window.YV.Graph`, `window.YV.LAYOUT_SETTINGS`). Data payloads use the
distinct globals `window.YV_SPEC` and `window.YV_ACT` so the boundary
between *code* and *generated data* is visible at a glance.

## Tunable spacing — top of `layout.jsx`

Most everything you'd want to nudge is exposed via the in-app Settings
drawer (⚙ in the header) — colors, gaps, padding, edge tails, neck
offsets, container dashes, brochure thumb size. Edits persist via inline
mutation of `window.YV.LAYOUT_SETTINGS`; reset restores defaults.

## How to run

Two paths, both fine:

- **Server (recommended).** `uv run yolovex serve` — opens an HTTP server
  at http://127.0.0.1:8765 with a 📷 upload button that runs a fresh
  capture on any image you pick and streams progress back via SSE.

- **Static.** Open `frontend/index.html` directly in a browser. The
  activations are whatever the last `yolovex build-assets` run wrote.

The static path works because `index.html` references `spec-data.js`,
`activations.js`, and the `.jsx` siblings with relative paths, so any
static-file viewer (or just `file://`) renders correctly.

## Attention prototype

`attention-prototype.html` is a throwaway standalone page for the ROADMAP
attention-visualization item. It loads `attention-prototype-data.js`, a compact
uint8 export of the C2PSA post-softmax attention tensor plus a matching preview
image.

Regenerate the payload after changing the capture contract:

```bash
uv run python scripts/export_attention_json.py
```

Then open `frontend/attention-prototype.html` directly, or serve the folder:

```bash
uv run python -m http.server 8766 --directory frontend
```
