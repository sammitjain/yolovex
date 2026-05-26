# yolovex frontend — implementation notes

The interactive explorer. Static HTML + React (via Babel-standalone in the
browser, so no build step). It reads two generated payloads:

- `spec-data.js` — `window.YV_SPEC`: deduplicated block specs, instances,
  and the data-flow edges. Produced by `yolovex.block_spec`.
- `activations.js` — `window.YV_ACT`: per-fx-node activations + Detect
  payload (boxes, per-class score maps) for one image. Produced by
  `yolovex build-assets` (or rewritten on each upload by `yolovex serve`).

In-Region layout is computed by **ELK** (`elkjs`); see ADR-0001/0006/0007. ELK
owns node placement + in-Region edge routing inside one expanded Region at a
time; `layout.jsx` owns the L1 (cross-Block) placement and edges.

## Files

| File | Role |
| --- | --- |
| `index.html` | Entry point. Loads React/Babel, the ELK bundle, the two payloads + content, and the jsx files (load order below). |
| `arch.jsx` | Data model. Joins `YV_SPEC.instances` + `edges` with a small **presentation** config (`PRESENTATION`: per-block `col`/`vpos`/`role`). Roles: Backbone 0–8, Neck 9–22, Head 23. Type/role palettes for light + dark. |
| `layout.jsx` | L1 pixel positions, Role-frame shapes, cross-Block edge paths. Columns are flow-order walks so expanded blocks slide neighbours aside. All tunable spacing lives at the top of this file. |
| `graph-sem.jsx` | Shared graph **semantics**: the pure transforms (`preprocessGraph`, `aggregateWithExpansions`, `classifySubkind`) exposed as `window.YV._graphSem`, plus node-sizing rules, region padding, and the sub-kind colour palettes. Layout-engine agnostic. |
| `expand-elk.jsx` | In-place expansion via ELK. `buildExpansionELK(idx, {flip, expansions})` (async — ELK lays out off-thread) turns one Block into a laid-out internal sub-graph. Reuses `_graphSem` for the semantic front-half; ELK does the geometry. |
| `graph-elk.jsx` | SVG render + pan/zoom/hover (`window.YV.Graph`). Clicking a Block toggles expansion; renders expanded Regions, their internal sub-graph, and ELK-routed edges. |
| `app.jsx` | App shell — header, flow play, Settings drawer, DetailPanel, DetectPanel, the 📷 upload flow (server mode), and the `BuildProgressOverlay`. |
| `vendor/elk.bundled.js` | ELK layout engine — UMD global `ELK`, loaded before the jsx. |
| `spec-data.js` | Generated. Architecture spec — committed alongside source. |
| `activations.js` | Generated. Per-image activations — regenerated on every `build-assets` / upload. |
| `design-spec.html` | Static design doc; not loaded at runtime. |

jsx load order in `index.html`: `arch.jsx`, `layout.jsx`, `graph-sem.jsx`,
`expand-elk.jsx`, `graph-elk.jsx`, `app.jsx`.

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

Prototype notes live in `ROADMAP.md`; keep this file focused on the current
frontend runtime.
