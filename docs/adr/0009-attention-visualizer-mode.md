# Attention visualizer: full-page mode reached from a canvas control

---
Status: accepted
Supersedes: ADR-0008 (UX section only — the data contract there still stands)
---

ADR-0008 settled *how* the post-softmax attention tensor is captured and shipped
to the frontend. The original UX idea inside that ADR — render an inline
"Attention instrument" inside the side panel while the user is looking at
C2PSA[10] — was prototyped in `frontend/app.jsx` against every C2PSA-typed
selection, and it didn't carry its weight: the toggle was too broad (any C2PSA
sub-node inherited it), context-switching inside a 380px-wide side-panel column
felt cramped against canvas navigation, and the standalone prototype at
`frontend/attention-prototype.html` (the validated reference for the UX) was a
full-page layout, not a panel widget. This ADR records the pivot.

## Decision

The Attention visualizer is a **canvas-level mode**, not a panel widget. It is
reached two ways and rendered the same way:

1. **A canvas control button** in the zoom-controls cluster
   (`frontend/graph-elk.jsx`), next to the theme/collapse/fit/zoom buttons.
   The button only renders when `window.YV_ACT?.attention` is present.
2. **A targeted affordance on the specific Attention sub-node** — when the
   user has selected `{ idx: 10, pathKey: '0_PSABlock/attn' }`, the side
   panel shows an "Open attention visualizer" button. The targeting comes
   from a `YV_CONTENT_OVERRIDES` entry keyed exactly
   `"10/0_PSABlock/attn"` with `openAttentionVisualizer: true`, so the
   affordance never accidentally bleeds onto other C2PSA sub-nodes or other
   C2PSA-typed blocks elsewhere in the network (currently none; a defensive
   guard for the future).

When opened, the visualizer **replaces the architecture view entirely** —
Graph, FlowOverlay, DetailPanel, SettingsPanel, BuildProgressOverlay are not
rendered while `attentionMode === true`. A "← Back to architecture" button in
the visualizer's top bar exits. (The app header remains visible because the
visualizer is overlaid inside `.app-main`; that's intentional — it keeps the
user oriented and Settings reachable when they exit.)

The visualizer UI is a React port of `frontend/attention-prototype.html`, so
the look the user already validated maps 1:1: image on the left, control rail
on the right (head buttons, per-query/global normalisation, play, color map,
overlay opacity, speed, row/col inputs, query/peak/weight/entropy metrics,
mini heat-grid).

### One opacity slider, no separate dim

The earlier inline-panel prototype exposed *two* compositing controls — `alpha`
(heatmap opacity) and `dim` (input-image darken). The user pushed back: the
two-control surface read as fiddly noise without clarifying anything the
heatmap couldn't carry on its own. The standalone prototype only has `alpha`
and that was strictly nicer to use. The visualizer drops `dim`; only `alpha`
survives. This is intentional and should not be re-added without user request.

## Why not a side-panel widget

- **Targeting was too broad.** Type-keyed on `C2PSA` means every sub-node
  inside C2PSA[10] inherits the same toggle/description. The user only wants
  the affordance on the one Attention node whose softmax is actually captured.
- **The side panel is for interpretation copy, not for full interactive
  instruments** (per ADR-0003). The Attention visualizer is closer in spirit
  to the Detect panel (`DetectPanel`) — a full mode reached from a node — than
  to the brochure/IO-strip widgets that share the side panel.
- **The prototype the user liked was full-page.** Reproducing it inside a
  ~380px panel column would not have matched what was validated.

## Why not a separate HTML route

Considered:`/attention.html` mounted alongside `/index.html`. Rejected:

- Two HTML routes means two `<script>` graphs, two Babel boot-up costs, two
  copies of asset-load logic, and the "Back to architecture" path becomes a
  navigation (losing graph/panel state) rather than a mode switch.
- The handoff that drove this work explicitly said "as a route/mode (not a
  separate HTML file long-term)."
- One SPA reading `window.YV_ACT` from one place is also the simpler upload
  story — uploads go through the main app's header `UploadButton`, the SSE
  pipeline rebuilds `activations.js` (with the new attention payload, per
  ADR-0008), and the visualizer reads the fresh payload on next open.

## Consequences

- `frontend/attention-prototype.html` + `attention-prototype-data.js` +
  `scripts/serve_attention_prototype.py` remain on disk as **the design
  reference** for the visualizer (and the place uploads go for an *offline*
  attention-only loop). They are no longer the productionised path; the
  in-app visualizer is. We do not need to keep them perfectly in sync with
  every visualizer change; if they drift far enough to be misleading, retire
  them rather than back-port.
- The visualizer reads `window.YV_ACT.attention` exclusively (ADR-0008's data
  contract). Anything new the visualizer needs (e.g. value tensors, per-head
  bias) goes through ADR-0008 amendments, not visualizer-local hacks.
- Side-panel content for the Attention node is the natural home for
  visualizer-mode copy. Other nodes that grow dedicated full-page modes in
  the future (e.g. a Detect "play with thresholds" mode) should follow the
  same pattern: a `YV_CONTENT_OVERRIDES` flag on the exact selection key, a
  small affordance in the side panel, an App-level `*Mode` boolean, and an
  overlay component swapped in for the canvas + panels.
- ADR-0008's UX section is superseded by this ADR. Its capture mechanism,
  quantisation, and `window.YV_ACT.attention` payload shape are unchanged.
