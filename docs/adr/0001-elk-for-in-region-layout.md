# Adopt ELK for in-Region layout (direction, not yet implemented)

---
Status: proposed — this records the *direction* we intend to move. The current
codebase is NOT migrated. We will validate ELK against the real requirements
with an experimental spike first, then migrate under git once proven.
---

## Context

The diagram's in-Region layout grew reactively: five hand-built placement
strategies (spine / staircase / fan-in / fan-out / skip), three node shapes, a
bespoke flat-bezier family, and an `INNER_PAD` "inset reconciliation" hack that
fakes compound nesting. This sprawl is the root of the alignment problems
(feature 1) and the edge-case maintenance burden (feature 4). The desired
behaviours — natural alignment, expansions that never break alignment,
dimension-annotated edges, and routing we don't hand-roll — are all standard
output of a layered (Sugiyama) graph-drawing engine.

## Decision

Adopt **ELK (`elkjs`)** as the layout/routing engine, **scoped to Region
internals**: layered layout, true compound/hierarchical nesting, edge labels
(fed from `YV_SPEC.shapes_by_node`), and bend-point edge routing.

The **top-level placement of the 24 Blocks stays hand-authored** — the
paper-faithful Backbone / L-shaped Neck / Head layout. Those are seed positions
ELK does not touch; ELK owns everything from the Block boundary inward.

`elkjs` ships as `elk.bundled.js`, a single UMD script loaded via `<script>`,
so this preserves the project's static-hosting requirement (see below). The SVG
renderer (`graph.jsx`) is kept; it is fed ELK's computed coordinates and bend
points instead of `layout.jsx` / `expand.jsx` geometry.

## Constraint being protected

The real requirement is **free static-hostable output**: the public build must
deploy as plain static files (GitHub Pages / Netlify / Cloudflare / HF static),
so anyone can use it at no cost. This is independent of two things people
conflate with it:

- **Build step vs not** is orthogonal — a bundler (Vite) still emits static
  files. The current "no build step" (Babel-in-browser) is a convenience, not a
  requirement; a build step is acceptable if it opens options.
- **Static vs server hosting** is the live-upload question: only the PyTorch
  on-demand capture (`yolovex serve`) needs a server. The public build can ship
  precomputed activations and stay fully static. (Tracked separately under
  feature 5.)

ELK-via-script-tag keeps even the no-build convenience; but adopting a build
step later would not violate the static-hosting requirement.

## The seed-position / reflow boundary

ELK does not know the hand-authored paper layout, so the responsibility splits:

- **Inside a Region** — ELK owns sibling reflow natively: expanding a Sub-node
  slides its siblings aside (compound layout). This is the painful part today.
- **Top level (24 Blocks)** — *we* keep the existing thin reflow pass
  (`layoutGraph`'s flow-order walk): it consumes each Block's measured size
  (Box height collapsed, **ELK-measured Region size** when expanded) and slides
  later Blocks / shifts right-hand columns. Paper fidelity and the current
  "neighbours slide aside on expand" behaviour are preserved; only the geometry
  *source* changes (ELK instead of hand-rolled `regionW/regionH`).

This boundary is the primary thing the spike must validate.

## Considered options

- **Hand-roll the Sugiyama phases** — rejected: reinventing a solved problem;
  doesn't scale as the fully-expanded graph approaches ~400 nodes (attention +
  Detect internals).
- **dagre** — rejected: flat only, no compound layout, weak ports/edge-labels.
- **React Flow (`@xyflow/react`) + ELK + a build step** — deferred on
  cost-and-need, NOT on tooling. A build step is acceptable (output stays
  static), so React Flow is not blocked. But ELK-alone already delivers all four
  target behaviours while *keeping the existing SVG renderer*; React Flow would
  mean rewriting `graph.jsx` wholesale, and its main bonus (draggable canvas) is
  explicitly parked. Revisit when drag/interaction needs justify the rewrite.

## Consequences

- Retires the five placement strategies, the three-shape special-casing, the
  hand-rolled bezier family, and the `INNER_PAD` inset hack.
- ELK's API is async (promise-based `layout()`); the render flow must adapt.
- Adds a ~1.4MB script dependency (acceptable for a load-once learning tool).
- Migration is staged behind git: spike → validate against features 1–4 → adopt.

## Spike status (update)

The spike lives in `frontend/index-elk.html` + `expand-elk.jsx` + `graph-elk.jsx`
(legacy `index.html`/`graph.jsx` untouched). ELK handles layering, crossing-min
ordering, compound nesting (stray/disappearing edges at depth: solved), and the
FPN-up flip. **One limitation surfaced:** ELK cannot left-edge-align the trunk
(center-based placement; no native option — confirmed offline + online, elkjs
issue #217). Resolution: keep ELK for ranks/order/nesting and add a
coordinate+routing post-process — see **ADR-0005**, which is the next
implementation step.
