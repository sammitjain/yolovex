# yolovex — Diagram & Layout

The shared language for the interactive architecture diagram: what the user
clicks, what unfolds, and how it's placed on the canvas. This glossary covers
the **diagram/layout domain** only (the canvas, blocks, expansion, placement).
Styling and side-panel content are separate domains, grilled separately.

## Language

### Graph hierarchy

**Block**:
A top-level unit on the canvas — one of the 24 things you see before expanding
anything. Mirrors how the model reads as a list of modules in PyTorch.
_Avoid_: layer (a Block may contain several layers), L1 node.

**Node**:
Any vertex in the graph, at any depth — a Block is a Node, and so is anything
revealed inside one. The generic graph/geometry word.
_Avoid_: using "node" to mean specifically a Block or specifically a Sub-node —
say which one.

**Sub-node**:
A Node revealed *inside* a Block when it's expanded (e.g. Conv → Conv2d / BN /
SiLU). Every Sub-node is a Node; it lives at depth ≥ 1.
_Avoid_: sub-component, internal node, sub-block, leaf module (use Leaf for the
atomic case).

**Leaf**:
A Node with no expandable internals — atomic. Orthogonal to depth: most Leaves
are Sub-nodes (Conv2d, SiLU, BatchNorm2d, MaxPool2d), but some top-level Blocks
are also Leaves (Concat, Upsample, and — for expansion purposes — Detect).
_Avoid_: leaf module, atom.

### Footprint

**Box**:
The fixed-size footprint a Node occupies while *collapsed* (`NODE_W × NODE_H`).
Every Node has a Box.
_Avoid_: tile, card, cell.

**Region**:
The content-sized space a Node occupies while *expanded*. Recursive: a Block's
expansion is a Region, and an expanded Sub-node's space is a nested Region.
_Avoid_: container, inner container (a nested Region, not a separate thing),
self-region.

### Roles & framing

**Role**:
The semantic grouping a Block belongs to — Backbone, Neck, or Head. Carries the
YOLO meaning (feature extractor / FPN-PAN fusion / detector) and drives palette
+ framing. The Neck additionally spans an L-shape because SPPF/C2PSA sit in the
backbone column.
_Avoid_: section, stage, group.

**Role frame**:
The drawn outline around every Block of one Role (rect for Backbone/Head, the
mirrored-L polygon for Neck). A visual cue and label only — it does not drive
placement (the `col`/`vpos` presentation data does).
_Avoid_: container, placeholder (that word is the fx graph input node).

### Layout model (layered / Sugiyama)

The diagram is a **layered graph drawing** (the Sugiyama framework — what dagre /
ELK / d3-dag implement), applied recursively to each Region (compound layout).
We adopt this vocabulary wholesale rather than inventing per-block patterns.

**Flow**:
The primary layout direction. Top-to-bottom by default; an FPN-up Region flips
it bottom-to-top. Edges run along the Flow.
_Avoid_: dataflow direction (reserve "dataflow" for the model's semantics).

**Layer**:
A horizontal band holding all Nodes at the same depth along the Flow (longest-
path rank). The code's `rank` is this.
_Avoid_: rank (code alias only), row, level (level is the retired fidelity term).

**Order**:
A Node's position within its Layer, chosen to minimise edge crossings.
_Avoid_: column index, slot.

**Port**:
A point on a Node's side (top / bottom / left / right) where an edge attaches. A
side may carry **several** Ports — at the L1/Block level there is one mid-side
Port per side (`leftPort` / `rightPort` / `topPort` / `botPort`), but inside a
Region a single side hosts one Port per attaching edge.
_Avoid_: anchor, connector.

**Port fraction**:
A Port's position along its side, as a fraction of the side length (0 = start,
1 = end; e.g. two ports on a top edge at 1/3 and 2/3). ELK distributes Ports at
even fractions by default; overriding them spreads the edges wider.
_Avoid_: port offset, port ratio (ELK's internal name), slot.

**Long edge**:
An edge spanning ≥2 Layers. Subsumes what used to be "skip" *and* "staircase
tap" — both are just long edges, routed identically.
_Avoid_: skip, skip lane, tap (when meaning the edge).

**Merge**:
A Node with in-degree ≥ 2 — two or more edges arrive at it (e.g. SPPF's `cat`,
the residual `add`). A graph-shape descriptor, not a layout strategy. Drives the
fan-in width boost (widen the Node so its inputs spread across one edge).
_Avoid_: fan-in (retired strategy term), join, sink.

**Split**:
A Node with out-degree ≥ 2 — two or more edges leave it. A graph-shape
descriptor. Drives the fan-out width boost. Distinct from the **split op** (the
`.split()` / `.chunk()` operation, e.g. in C2PSA): a split op is *a* Split, but
not every Split is a split op.
_Avoid_: fan-out (retired strategy term), branch point; and don't conflate the
shape (Split) with the operation (split op).

**Bend point**:
A waypoint on a routed edge where it changes direction (what the renderer draws
as a path vertex). Long edges route through bend points instead of the retired
skip-lane / staircase machinery. ELK computes these; internally ELK uses "dummy
nodes" to do it, but those stay inside the engine — we only consume bend points.
_Avoid_: dummy node (ELK-internal only), skip lane, waypoint (informal synonym).

**Compound layout**:
Laying out a Region's children with the layered algorithm, then sizing the
Region to fit so it acts as one Node in its parent's layout. The recursion that
makes nesting and flow compose.
_Avoid_: nested layout, hierarchical layout (acceptable synonyms).

### Styling

**Token**:
A named themeable style value (color, radius, shadow) whose single home is a CSS
custom property. Has one value per Theme.
_Avoid_: var, constant, setting (LAYOUT_SETTINGS is the runtime layer, not the
source).

**Theme**:
A complete variant of every Token — Light or Dark — selected via the
`data-theme` attribute. Both variants are always maintained in tandem.
_Avoid_: mode, skin.

**Palette**:
A coherent group of Tokens addressing one axis of the graph: by node *type*
(Conv, C3k2…), by *Role* (Backbone/Neck/Head), or by *subkind* (cat, split,
arith…). A grouping of Tokens, not a separate color store.
_Avoid_: color scheme, theme (Theme is the light/dark axis, Palette is the
by-category axis).

### Side-panel content

**Type card**:
The panel layer describing what *kind* of block the selected Node is — YOLO role
first, then the friendly Intuition. Type-level (keyed by class).
_Avoid_: explainer, about-box.

**Intuition**:
The friendly, conceptual answer to "what is this block?" — type-level, plain
language. The voice to preserve.
_Avoid_: description (too generic), summary.

**Interpretation**:
Guidance for reading the *activation* — "what am I looking at, is anything
notable?" Distinct from Intuition (which explains the block, not the picture).
Hand-authored, keyed per-block-position with a per-type fallback. Must be
**image-robust**: describe tendencies and what to look for (hedged), never
assert specific content that only holds for the bundled sample image.
_Avoid_: analysis, intuition (Intuition explains the block; Interpretation reads
the activation).

**Activation view**:
The instance-level panel layer — IO strip, channel brochure, statistics — for
the selected Node on the current image.
_Avoid_: channel stack (that is one component of it).

## Flagged ambiguities

- **"Container" is retired.** It was used for Regions, Role frames, and Sub-node
  nesting paths. Resolution: expansion space → **Region**; role grouping →
  **Role frame**; a Sub-node's nesting path → **Region path** (impl rename).
- **"Placeholder"** stays reserved for the fx graph **input node** — never reuse
  it for Role frames.
- **The five layout strategies are retired.** `spine` → the Flow column;
  `staircase`/`tap`/`STAIR_FRAC` and `skip`/`skipLane` → **Long edge** + **Dummy
  node**; `fan-in`/`fan-out` as *layout strategies* are retired — the Sugiyama
  model absorbs them. The in/out-degree ≥ 2 node *shapes* are now named **Merge**
  and **Split** (see Layout model); they carry only a width-boost *sizing* rule
  fed to ELK, not a bespoke placement strategy.
