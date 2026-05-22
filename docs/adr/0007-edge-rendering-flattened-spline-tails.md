# Edge rendering: ELK spline routing with flattened port tails

---
Status: accepted (resolves the rendering choice deferred by ADR-0006)
---

ADR-0006 kept ELK as the source of node placement *and* edge routing, but left
the **render style** open: "Edge rendering style (orthogonal vs curved) is still
ours to choose at draw time from ELK's bend points." This ADR records that
choice for the in-Region (ELK) edges and the cross-Block (L1) edges.

## Decision — in-Region edges (`frontend/expand-elk.jsx`)

Use ELK's `elk.edgeRouting: 'SPLINES'` and render a **hybrid** path: keep every
interior control point ELK produced (its node-avoidance routing — the part that
minimises crossings) and reshape **only the first and last segment** so the edge
leaves the source and enters the target **dead-vertical** (the in-Region Flow is
vertical). This gives soft, ELK-routed curves with straight, perpendicular
"tails" at the ports.

`_elkFlatTailSpline(pts)` does this. Key facts that shaped it:

- **ELK SPLINES output format.** Points are `start` followed by groups of 3
  `(control, control, anchor)` per cubic, with the final anchor == `end`. Any
  leftover trailing points form a **straight polyline tail** into the port (one
  cubic that veers over, then e.g. a vertical drop). So the emitter draws cubic
  groups *and* the trailing remainder as `L` segments. **Dropping the remainder
  leaves the path short of the endpoint — the edge dangles.** (This was a real
  regression; the remainder loop is load-bearing.)

- **Tail flattening = override two control points.** Set the first cubic's
  leading control to `{startX, startY ± exit}` and the last drawn control to
  `{endX, endY ∓ entry}`, forcing vertical tangents. Tail lengths
  `ELK_TAIL_EXIT` / `ELK_TAIL_ENTRY` are the dials; they are clamped against the
  edge's vertical span.

- **When to flatten (the adaptive guard).** A single cubic (`n === 4`) has no
  interior routing points, so flattening either tail just yields a clean S —
  **always safe**, even when the target is a wide Merge whose port is far
  off-axis (e.g. C3k2's `cv2 → add`). For longer edges (`n ≥ 5`) ELK has
  interior bend points, usually a **side lane** it routed into to avoid nodes;
  only flatten the tail there if ELK's own tangent is already near-vertical
  (`|dx| ≤ |dy|`). Forcing a lane edge vertical fights the routing and bulges
  the curve concavely (observed on C3k2's chunk→cat skip exit). Consequence: a
  long edge that enters a wide Merge at a steep angle (chunk→cat) keeps ELK's
  angled entry — accepted for now.

  Note: short edges between vertically-aligned nodes are already straight
  vertical lines (every ELK point shares one x); tail length has **no visible
  effect** on them — there is no horizontal offset to flatten.

## Decision — cross-Block (L1) edges (`frontend/layout.jsx`)

L1 edges are not ELK's (ADR-0006). They already render as flat-tail beziers
(`flatBezier` / `flatBezierVertical`). One refinement landed here: an expanded
block's entry node that **also has an internal predecessor** is a Merge of the
external block input + an in-Region input (e.g. SPPF's residual `add`, fed by
both `x` and `cv2_act`). Its external input is routed to the node's **left
port** and drawn with `flatBezierVtoH` (vertical out of the source, horizontal
into the left port) so the two inputs of the Merge read as distinct, and a
circular `add` is fed into its face rather than having a line drop onto its top
edge. `targetPorts` tags side ports with `horiz` to select the bezier.

## How this was verified (browser introspection)

ELK's *output* (not just its input, cf. the `window.__elkFail` note in
CLAUDE.md) was inspected live: drive the running Space with the Claude preview
tools, then in `preview_eval` wrap `window.YV._elk.layout` to capture the laid-
out `root`, call `window.YV.buildExpansionELK(idx, {expansions:[...]})`, and walk
the captured tree computing the flatten decision per edge. This pinned the exact
edges that were/weren't flattened (e.g. confirmed `cv2 → add` was n=4-but-angled,
and that SPPF's maxpool taps were already flattened) from real data instead of
hand-built graphs.
