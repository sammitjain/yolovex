# ELK owns node placement and in-Region edge routing

---
Status: accepted (supersedes ADR-0005)
---

ELK's `layered` algorithm produces our in-Region layout end-to-end: layer
ranking, crossing-minimised in-layer order, compound nesting/sizing, **node
placement (X)**, and **edge routing** (orthogonal bend points). We do **not**
post-process its output — no custom coordinate pass, no hand-rolled edge paths.
This reverses ADR-0005, which proposed owning X (left-pin) and routing.

## Why

We spec'd and prototyped the left-align post-process and found it both
unnecessary and harmful: ELK's native **center-aligned** placement is
aesthetically fine, and ripping out ELK's routing to hand-draw edges discarded
working bend-point routing and re-introduced retired bespoke logic. Critically,
because ELK re-solves the whole layout (including edge routing) whenever its
inputs change, staying inside ELK means branch-order and spacing tweaks
**re-route automatically** — a post-process would have to redo that by hand.

Much of what used to need bespoke code now falls out of ELK for free: the SPPF
maxpool "staircase", nested-Region framing, indentation, and long-edge lanes are
emergent, not special-cased (this retires the strategies already flagged in
CONTEXT.md).

## What we control instead — ELK *inputs*, not its output

- **Node placement strategy:** `BRANDES_KOEPF` with `bk.fixedAlignment: LEFTUP`
  — the corner that gives the desired branch staircase. (LEFTUP's staircase
  reads top-right→down rather than the mirror we'd ideally want; accepted.)
- **In-layer left/right order:** `considerModelOrder.strategy: NODES_AND_EDGES`
  so a fan-out's first-declared (fx-execution-order) branch sits on the left.
- **Branch spread:** per-node width boosts keyed on degree — fan-in
  (`WIDEN_STEP_IN`) and fan-out (`WIDEN_STEP_OUT`) — so multiple edges spread
  across a node's edge instead of overlapping.
- **Port spacing:** `elk.spacing.portPort` to widen the gaps between multiple
  edges meeting one node side.

These are all small, declarative knobs fed *into* ELK; ELK does the geometry.

**Apply them at every nesting level.** Under `hierarchyHandling: INCLUDE_CHILDREN`,
spacing/placement options set on the root do **not** propagate into child
containers — each nested Region falls back to ELK defaults. So the knobs above
are spread into `root` *and* every container's `layoutOptions`; only `direction`
and `padding` are set per-node. (Forgetting this is why tuning `nodeNode` on the
root once appeared to "do nothing" to nodes inside an expanded sub-block.)

**One exception — `considerModelOrder` on boundary-crossed containers.** elkjs
throws (`Cannot read properties of undefined (reading 'a')`) if
`considerModelOrder` is set on a container whose boundary a cross-hierarchy edge
crosses — an edge declared at an ancestor LCA with an endpoint inside the
container (true for any value of the strategy, not just `NODES_AND_EDGES`). This
is common: every expanded sub-block connects to the outside, and an edge-less
container is just the degenerate case. So the option is stripped from every
boundary-crossed (and edge-less) container after edges are assigned, while the
root keeps it (its boundary can't be crossed). The cost: a nested container with
internal branching that also connects outward loses its first-branch-left
ordering — accepted, since the alternative is an ELK crash that collapses the
whole block, and top-level branch ordering (the visible SPPF/C2PSA staircase) is
unaffected because the root is never stripped.

## Consequences / boundary

- ELK only lays out the **inside of one Region at a time**. It has no knowledge
  of sibling Blocks, so **cross-Block (L1) edges are NOT ELK's** — they remain
  routed by `layout.jsx`. Improving a specific L1 edge (e.g. the residual into
  SPPF's `add`) is a `layout.jsx` concern, tracked separately, not an ELK knob.
- Edge *rendering* style (orthogonal vs curved) is still ours to choose at draw
  time from ELK's bend points; that is a presentation decision, not a placement
  one.
