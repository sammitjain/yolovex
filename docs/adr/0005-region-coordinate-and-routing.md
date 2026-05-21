# Region coordinate assignment + edge routing (post-process on ELK)

---
Status: proposed — agreed strategy; to be IMPLEMENTED IN A FRESH SESSION against
this spec. ELK integration (layering/ordering/nesting) already works in
`frontend/expand-elk.jsx`; only the coordinate + routing pass below is pending.
---

## Context

ELK's `layered` algorithm center-aligns nodes on the cross-axis and offers **no
native left-edge alignment** — confirmed offline (every nodePlacement strategy +
`fixedAlignment` corner + `alignment`/`compaction` options leave the trunk
centered) and online (elkjs issue #217 asks for exactly this and is unanswered;
`contentAlignment` only positions a node's whole content block inside a larger
parent). The desired look — **trunk pinned left, branches/taps to the right** —
is what makes an expansion feel like a *natural* in-place reveal (the Region's
top-left maps to the parent Block's top-left).

## Decision

Split responsibility:

- **ELK owns** layer assignment (ranking), crossing-minimized in-layer ordering,
  and compound/hierarchical nesting + sizing (the hard, previously-hand-rolled
  parts).
- **We own** the final **X-coordinate assignment** (left-edge alignment) and
  **edge routing**, applied as a post-process to ELK's output.

This is NOT a retreat from ELK — we consume its ranks + order + nested sizing and
only restyle coordinates/routing to the target aesthetic. Edge routing becoming
ours folds naturally into the deferred curved-edge pass (ADR-0001 #4).

## Spec — alignment

- Group ELK's laid-out nodes by layer (same rank / same cross-flow band).
- For each layer, align the **left edge** of the leftmost node to the Region's
  left content margin (`REGION_PAD_X`); lay siblings out left→right from there.
- Single-node layers → a straight left-pinned vertical trunk. Multi-node layers
  (true parallel branches) start at the margin and extend rightward.
- A widened merge node (e.g. `cat`) keeps its left edge on the trunk and grows
  rightward.

### Note 1 — reverse the in-layer order before aligning
ELK's crossing-min order currently renders SPPF's taps **right-to-left as you
descend** (first MaxPool far right, later ones stepping left) — counterintuitive.
When assigning X, **reverse the offset direction** so taps read **left→right with
depth**. Likely a one-line flip of how we sort/offset same-layer (or tap) nodes;
verify on SPPF.

## Spec — routing

- **Trunk edges** (consecutive layers, both left-pinned) → straight vertical.
- **Long / tap edges** (e.g. `cv1`→`cat` spanning the maxpools; skip edges) →
  exit the source, step into a right-side lane, run down, re-enter the target.
- Essentially the legacy spine-left / lanes-right look — that picture was always
  the goal; only the legacy *code* was the problem.

### Note 2 — vertical edges may curve, and that's fine
When a source node is wider than its target (left edges aligned, centers differ),
the connecting edge will not be perfectly vertical and may curve. This is
acceptable and matches the previous implementation's behaviour.

### Note 3 — SPPF residual `add` (feedback item)
The residual `add` at the tail of SPPF needs a long curve coming from the
previous block's last layer. That long edge may interfere with the placement of
the expanded SPPF's first `Conv`/`cv1` node. Implement, then expect a feedback
pass on exactly how the `add` and its incoming long edge resolve.

## Implementation pointers

- Lives in `frontend/expand-elk.jsx`, AFTER `await _elk.layout(root)` and BEFORE
  building `subNodes` / `subEdges`.
- Re-derive inner-container bboxes (`containerAbs`) from the SHIFTED node
  positions, not ELK's, so nested Region frames still wrap their children.
- Re-route edges from final node positions (replaces consuming ELK's
  `sections`/bendPoints); keep `entryNodes`/`exitNodes` as the external
  connection ports (layout.jsx reads them).
- Keep the contract identical: `{subNodes, subEdges, innerContainers,
  entryNodes, exitNodes, regionW, regionH, flip}`.

## Current spike state (for the fresh session)

Working in `frontend/expand-elk.jsx` + `graph-elk.jsx` + `index-elk.html`
(legacy `index.html` untouched):
- ELK layering + compound nesting + LCA-placed edges: working (stray/disappearing
  edges at depth — SOLVED).
- Merge-node widening by in-degree (`WIDEN_STEP`): in place; fan-out widening
  intentionally removed (kept maxpools uniform).
- `flip` (FPN-up col-1) via ELK `direction: UP`: confirmed correct.
- Edge routing: currently ELK ORTHOGONAL bend-points (placeholder until this
  spec's custom routing lands).
- LEFT ALIGNMENT: NOT done — this ADR is the plan for it.
