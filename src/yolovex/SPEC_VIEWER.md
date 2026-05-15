# Block Spec Viewer

A derived, container-transparent view of every block in the model at multiple
fidelity levels. Everything is computed from `torch.fx` — there are no
hand-coded per-block templates.

## Pipeline

```
PyTorch model
   │
   ▼
src/yolovex/block_spec.py
   │  for each top-level block:
   │    1. torch.fx.symbolic_trace(block)
   │    2. ShapeProp → annotate each fx node with output shape
   │    3. compute_visible_path(node) for every fx node
   │    4. build_path_class_map(block) — class lookup for every visible path
   │    5. graph_topology_hash(graph) — dedupe key
   ▼
out/specs/library.json            (read-only source of truth)
frontend/spec-data.js             (same content, JS-wrapped as window.YV_SPEC)
   │
   ▼
frontend/spec-viewer.html  +  spec-viewer.jsx
   │  per-render:
   │    1. preprocessGraph     — hide shape ops + get_attr, collapse getitem
   │    2. aggregateAtDepth    — group nodes by visible_path prefix,
   │                             attach containerPath per aggregated node
   │    3. detectStaircases    — strict-chain merges (SPPF)
   │    4. detectFanInMerges   — parallel-branch generalization
   │                             (e.g. C3k inner cat with cv2 bypass)
   │    5. autoLayout          — layered top-down, rank-by-rank placement
   │                             tracking actual x, skip lane assignment,
   │                             target alignment, fan-in widening,
   │                             center-anchoring of plain spine nodes
   │    6. computeContainers   — bbox-nest hierarchy of parent containers
   │    7. SVG render          — containers behind, then edges, then nodes
```

Run after a model / weights change:
```bash
uv run python -m yolovex.block_spec
open frontend/spec-viewer.html
```

## Visible path — the one trick that makes everything work

Raw fx paths include `nn.Sequential` and `nn.ModuleList` segments that aren't
architecturally meaningful. **visible_path** is the same path with container
segments removed; container children inherit a `<index>_<ClassName>` suffix
to stay distinguishable (e.g. `0_Bottleneck`, `1_PSABlock`).

| fx node | raw path | visible path |
|---|---|---|
| SPPF.cv1.conv | `cv1.conv` | `['cv1', 'conv']` |
| C2PSA.m.0.attn.qkv (PSABlock at m.0) | `m.0.attn.qkv` | `['0_PSABlock', 'attn', 'qkv']` |
| C2PSA matmul inside attn | (from `nn_module_stack`) | `['0_PSABlock', 'attn', '_op', 'matmul']` |
| Block 22 m.0.0.cv1 (Bottleneck inside Sequential inside ModuleList) | `m.0.0.cv1` | `['0_Bottleneck', 'cv1']` |

Ops (`call_function`, `call_method`) attribute to their **call site** via
`node.meta['nn_module_stack']`, so a `matmul` inside attention shows up
inside attention's group, not floating at the top level.

For `call_module`, we use the **call site** (`nn_module_stack[-2]`) rather
than `node.target`. This matters for shared modules — Ultralytics has one
`nn.SiLU()` instance reused across every Conv block; without this, all
three calls hash to the same visible path and create cycles in the
aggregated graph.

## Fidelity levels

- **L1** — single block node + I/O (3 visual nodes total).
- **L2** — one level of container-transparent peeling. SPPF expands to
  `cv1, m×3, cat, cv2, add`; C2PSA expands to `cv1, split, PSABlock, cat,
  cv2`; Conv block expands to `conv, bn, act`.
- **L3** — one more level. `cv1` → `conv2d/bn/silu`; PSABlock → `attn / ffn
  Convs / internal residuals`.
- **L4+** — deepest detail. Attention internals, shape ops, etc.

The rule: `L_n` shows everything whose visible-path depth ≤ `n - 1`.
Containers (`Sequential` / `ModuleList`) are transparent — they don't count
as a depth level. `n` is bounded per-block by where the visible hierarchy
saturates (SPPF caps at L3, C2PSA at L4+).

## Op visibility

`preprocessGraph` runs before aggregation:
- **Non-tensor nodes** (ShapeProp shape `=== null`) — dropped entirely along
  with their incident edges. These are fx artifacts: `x.shape` access, integer
  B/C/H/W extraction, scalar arithmetic feeding `view`/`reshape` size args.
  Without this, Attention's L4 view fills with scalar-args-as-skip-arrows.
- `getitem` — **always** collapsed into an edge label (the index it pulls).
- `get_attr` — **always** hidden (it's metadata). Edges over it are reconnected.
- `view / reshape / permute / squeeze / unsqueeze / contiguous / transpose / flatten`
  — **always shown**, as small dashed boxes. They're real intermediate states
  a learner benefits from seeing, especially in Attention. (`opVisibilityForLevel`
  currently returns `{hideShapeOps: false, hideGetAttr: true}` at every level.)
- Everything else (cat, chunk, add, mul, matmul, softmax, …) — kept at every
  level. These encode real dataflow.

Note: scalar **constant** operands have no fx node at all — fx inlines Python
literals directly into `node.args`. So e.g. Attention's `attn * self.scale`
renders as a `×` with a single tensor input; the scale (`1/√d ≈ 0.1768`) lives
in the op's args, not as a drawable node. This is faithful to the trace, not a
bug — see Known limitations.

## Layout primitives

Several patterns recognise themselves automatically.

### Staircase (strict chain)

A merge (`cat`) whose predecessors form a **complete chain** — each tap's
only outgoing edge in the pred-set goes to the next tap, and the last tap
goes to the merge. Same idiom as `layout-l2.jsx`. SPPF is the canonical case
(`cv1 → m → m_1 → m_2 → cat`). Taps render on consecutive rows, each shifted
right by `STAIR_FRAC * NODE_W`; the merge widens to span them all; tap→merge
edges are straight verticals; tap→next-tap edges are right-angle beziers.
If the chain doesn't span every predecessor, this isn't a staircase — falls
through to fan-in or skip routing.

The chain check is strict on **both** sides: each non-start tap's *only*
predecessor must be the previous tap. Without the incoming check, a
residual-bypass pattern is mistaken for a staircase — e.g. C3k2's
`chunk → add → cat` where `add` also takes the Bottleneck output. That
mis-detection gave `add` a `STAIR_FRAC` offset while `chunk → add` was still
routed as a skip, leaving the skip line dangling. A real staircase tap
(SPPF's maxpools) is fed *solely* by its predecessor in the chain.

### Fan-in merge (parallel-branch generalization)

A `cat` whose predecessors include at least one **displaced tap** — a tap
with rank-gap ≥ 2 to the merge AND sitting in a multi-node row at its rank
(packing has pushed it off the spine column). Canonical case: C3k's inner
cat fed by `1_Bottleneck` (adjacent, end of chain) AND `cv2` (rank-3 bypass
packed alongside `cv1`).

When the trigger fires we widen the merge to span all the tap-centers and
render each pred→merge edge as a pure vertical drop. When no tap is
displaced (e.g. C2PSA's outer cat — both `split` and `'+2'` are alone in
their rows, both on the spine column), fan-in does **not** trigger and the
existing skip routing handles the long edge cleanly. This discriminator is
why we don't break C2PSA / SPPF / outer-C3k2 cats that already looked good.

### Skip edges with lanes

Any edge with rank-gap ≥ 2 that's NOT a staircase-internal edge and NOT a
fan-in tap edge. Renders as a vertical line dropping from the source's
bottom-left at `source.x + SKIP_OFFSET + lane * SKIP_LANE_W`. Two extras:

- **Lane assignment** — within a source-group (multiple skips sharing one
  source, e.g. C2PSA's `split → cat` and `split → '+1'`), the longest skip
  gets lane 0 (closest to source's left edge); shorter skips fan rightward
  at `+ lane * SKIP_LANE_W`. Prevents coincident vertical lines.
- **Nesting depth** — each skip's depth = count of other skips strictly
  containing it. Per-rank intermediate shift = `SKIP_SHIFT + depth * LANE_W`,
  so an inner skip's intermediates clear both the outer and the inner line.

### Target alignment + actual-x tracking

When a node is the **target** of a skip, its preferred x = `source.actualX +
lane * LANE_W`, so the skip line lands cleanly on the target's top edge
(no hanging arrow gap). We track `positionedX` per node during rank-by-rank
placement and use the source's *actual* x — important when packing has
displaced the source off its nominal column.

### Center-anchored spine

Nodes are positioned by their **left edge** `x`, but edges are drawn
**center-to-center**, and node widths are not uniform (`NODE_W` 150 for
modules/ops/cat, `SMALL_W` 90 for shape ops, `2*ARITH_R` 36 for arith
circles). A chain of mixed-width nodes all sharing the same left-edge `x`
therefore renders as a zigzag — its centers don't line up.

Fix: a **plain** spine node (the `else` placement branch — no incoming skip,
not a staircase tap, not a fan-in merge) that is *narrower* than `NODE_W` is
centered within the nominal `NODE_W` column: `x += (NODE_W - w) / 2`. Skip
targets keep left-edge semantics (the skip line lands at a precise x and the
target must not move); staircase taps and fan-in merges have their own
explicit x math.

**Uniform column slots.** Multi-node rows pack into uniform `NODE_W`-wide
slots: each peer's `slotLeft` is its preferred x clamped past the previous
peer's slot, the node is centered *within* its slot, and `curX` advances by
`max(actualWidth, NODE_W) + COL_GAP` — the *nominal* column width, not the
node's actual width. Without this, a narrow peer (an arith circle, w=36) on
one row advances `curX` less than a full-width peer on the next, so a parallel
branch packed to their right lands at a different x row-to-row — a zigzag
(observed on block 22 L5's `attn.pe` conv/bn/act, packed beside the
matmul / `×` / softmax attention spine).

### Row-sort heuristic

Within a multi-node row, peers are sorted by:
1. Preferred-x ascending (already-aligned nodes left).
2. `fanOutGap` ascending (low gap = main-path → leftmost; high gap =
   skip-branch peer → right).

This keeps the chain column on the spine while letting skip-branch peers
swing rightward.

### Arith ops

`add`, `mul`, `sub`, `truediv` render as small circles with the operator
glyph instead of rectangles.

### Parent containers

Every aggregated node carries a `containerPath` — the visible-path prefix
that names its enclosing parent. `computeContainers` collects unique
containerPaths plus all their ancestor prefixes (including `[]` for the
block-root), then computes bboxes in **depth-descending** order: each outer
container's bbox = union of its already-computed inner containers + its own
pad. Render order is shallow-first so deeper containers paint on top.
Result: an L3 view of C2PSA shows the outer `C2PSA` box wrapping everything,
with an inner `PSABlock` box wrapping just `attn`, `'+1'`, `ffn`, `'+2'`.

The container label sits at the box's **top-right** corner (`textAnchor=end`),
keeping it clear of the node column and edge curves on the left.

## Node labels

Nodes are labelled with their **type**, not their fx instance name — the class
for `call_module` (`Conv2d`, `BatchNorm2d`, `SiLU`), the op for
`call_function` / `call_method` (`fn:cat`, `.softmax()`), and the resolved
class for aggregated module groups (via `path_classes`). The fx name
(`m_0_attn_qkv_conv`) is noise for a learner; the type is what teaches. The
output shape still renders as a small line at the node's bottom edge.

## C3k2 fx-trace patch

C2f (parent of C3k2) writes its forward as:
```python
y = list(self.cv1(x).chunk(2, 1))
y.extend(m(y[-1]) for m in self.m)
return self.cv2(torch.cat(y, 1))
```
The generator expression hides each `m(y[-1])` call inside its own frame so
fx never sees the per-iteration tensor ops. `_traceable_c2f_forward` is a
drop-in replacement that uses an explicit `for` loop over `self.m` — fx
unrolls it over the (statically-sized) ModuleList. `_trace_with_patches`
builds a one-off subclass with the override and temporarily swaps the
instance's `__class__` for the duration of the trace, then restores. Same
output graph semantically. Unlocks all 8 C3k2 instances (blocks 2, 4, 6, 8,
13, 16, 19, 22) — including block 22's YOLO26 `Sequential([Bottleneck,
PSABlock])` injection.

## Per-block fidelity behaviour, observed

| Block | L1 | L2 | L3 | L4+ |
|---|---|---|---|---|
| **Conv** (0,1,3,5,7,17,20) | `Conv` | `conv → bn → act` (saturates) | — | — |
| **C3k2** c3k=False, n=1 (2,4) | `C3k2` | `cv1 → chunk → Bottleneck → cat → cv2` | Bottleneck residual surfaces | — |
| **C3k2** c3k=True, n=1 (6,8,13,16,19) | `C3k2` | `cv1 → chunk → C3k → cat → cv2` | C3k internals: `cv1, cv2 (bypass), Bottlenecks, cat, cv3` — fan-in widening on inner cat | further Conv/Bottleneck peel |
| **C3k2** attn=True (22) | `C3k2` | `cv1 → chunk → Bottleneck → PSABlock → cat → cv2` (Sequential transparent) | PSABlock internals appear alongside the Bottleneck | attn/ffn peel — same as C2PSA |
| **SPPF** (9) | `SPPF` | `x → cv1 → m → m₁ → m₂ → cat → cv2` staircase | cv1/cv2 expand to `conv/bn/act` | — |
| **C2PSA** (10) | `C2PSA` | `x → cv1 → split → PSABlock → cat → cv2` | PSABlock peels to `attn`, `'+1'`, FFN, `'+2'` with nested container | attn/ffn expand to qkv/proj/softmax — in progress |
| **Upsample** (11,14) | `Upsample` | (saturates immediately) | — | — |

## Known limitations

- **Detect (block 23)** — fx fails on it; not in the block dropdown.
  Separate design needed.
- **Concat blocks (12,15,18,21)** — top-level Concat takes a list of tensors
  as input, which fx can't trace directly. Trivially fixable when we need
  it.
- **L4+ for C2PSA / block 22** — Attention internals (qkv reshape, softmax,
  proj) generate many shape ops and small intermediate tensors. Layout
  needs further thought; see the design discussion outside this doc.
- **Cosmetic spacing** — horizontal offsets between containers and adjacent
  nodes could use tightening; tracked separately. The only differential
  right-shift lever today is `SKIP_SHIFT` (+ `SKIP_LANE_W` for nested skips),
  which pushes skip-straddled content — i.e. expanded container interiors —
  away from the regular node column. `MARGIN_PX` shifts everything uniformly.
  There is no per-container offset knob; containers are pure bounding boxes
  around their nodes, so a container can only be moved by moving its nodes.
- **Scalar constants not shown** — fx inlines Python literals into `node.args`
  rather than creating nodes, so constant operands (Attention's `* self.scale`,
  `transpose`'s dim args, etc.) have nothing to draw an edge from. The `×` for
  the attention scale therefore shows a single tensor input. Faithful to the
  trace; would need synthetic const-nodes (or an arg label) to surface.

## Files

- `src/yolovex/block_spec.py` — library builder, visible-path computation,
  CLI entry (`python -m yolovex.block_spec`).
- `src/yolovex/introspect.py` — older per-block tracing utility, still used
  for `_shape_of`, `capture_block_input_shapes`, `structural_signature`.
- `frontend/spec-viewer.html` — the page.
- `frontend/spec-viewer.jsx` — aggregation + layout + render. Reads
  `window.YV_SPEC` from `spec-data.js`.
- `out/specs/library.json` — generated, gitignored. Same content as
  `spec-data.js`.
- `frontend/spec-data.js` — generated, JS-wrapped library.
