# The side panel is an interpretation aid, not a reference

---
Status: accepted — the content model below is settled. Two layers exist today
(Type card, Activation view); Interpretation and the source-link refinement are
on ROADMAP.md.
---

## Context

The side panel runs on two content axes — Type-level (keyed by class: tagline,
intuition, technical, shape, yolo26, refs) and Instance-level (`YV_ACT`: IO
strip, channel brochure, raw min/max/mean/std). It hands the learner the
activation images plus raw statistics. The risk it must avoid is becoming a
**PyTorch reference** — reproducing textbook definitions and source code the
upstream docs already own.

## Decision

The panel is an **interpretation aid, not a reference.** Concretely, it carries
exactly these content elements, grouped in three layers:

**1. Type card** — what this *kind* of block is (keyed by class):
- **tagline** — one-line orientation under the title.
- **Intuition** — the conceptual hook: what this element does and its role in
  the architecture, whether YOLO-specific or generic deep-learning. The asset;
  keep its voice.
- **Theory** (`technical`) — the rigorous counterpart to Intuition: this model's
  concrete mechanics and parameters (kernel/stride, channel splits, residual
  conditions, where it sits in the net). Written **in the model's context**,
  never as a free-floating textbook definition. When there is nothing
  model-specific to say (structural wrappers like `Identity`/`Sequential`),
  omit it and rely on the source link.
- **YOLO26 note** (`yolo26`) — anything noteworthy for this block/leaf *in the
  YOLO context*. Optional; present where there's a genuine novelty.
- **shape** (optional) — a one-line tensor rule-of-thumb (e.g. "halves H,W,
  doubles channels").
- **Source link** — a per-class deep link to the upstream definition
  (Ultralytics for the YOLO blocks — `block.py` for the composites, `conv.py`
  for `Conv`/`DWConv`/`Concat`, `head.py` for `Detect`; PyTorch docs for the
  stdlib leaves). Replaces curated `forward()` source (see below).

**2. Activation view** — the instance-level layer, the headline value:
- **IO strip** — input → output thumbnails (a Merge shows `in1 + in2 →
  out`). Core to "watch the tensor transform."
- **channel brochure** — the grid of per-channel activation slices. The main
  thing the tool exists to show.
- **statistics** — raw min/max/mean/std summary. Kept (cheap; already computed);
  may be demoted/toggled later if it proves low-value.

**3. Interpretation** (new, not yet built) — *what to look for* in this
activation and whether anything is notable. Distinct from Intuition (which
explains the block, not the picture). **Hand-authored**, tightly scoped, and
**image-robust** — phrased as tendencies / what-to-look-for, since users upload
arbitrary images; never assert content that only holds for the bundled sample.
Scope incrementally: **L1 canvas blocks first**, sub-nodes/leaves later. The
attention-visualization work (`scripts/viz_attention.py`) feeds the
Interpretation notes for the attention blocks (C2PSA / Attention / PSABlock).

## On `forward()` source — dropped (reverses the earlier intent)

We considered curating each class's `forward()` source onto the panel and built
a pipeline for it (`content_build.py` → `YV_FORWARD`). **Dropped**, for two
reasons:

1. **It's incomplete.** A `forward()` body doesn't show the submodules built in
   `__init__`, so any on-panel snippet is a half-truth.
2. **It's redundant with the canvas.** The wiring of submodules into a
   computation — the whole job of `forward()` — is exactly what the
   expand-to-reveal canvas already draws. On-panel code would restate the canvas,
   less completely. The wiring *insight* is carried in prose by **Theory**, which
   also folds in the `__init__` params the bare body can't show.

Replacement: a **source link** per class (above). If a learner wants the code,
point them at the always-complete, always-current upstream file rather than
hand-maintaining a copy.

## Considered / deferred

- **Auto per-instance interpretation heuristics** (e.g. flag "mostly inactive,
  low std" from existing stats) — deferred. Interpretation is hand-authored
  first; heuristics are a possible later augmentation, kept honest if added.
- **A richer IO-strip rendering** than the current "in → out" row — wanted,
  low priority (on ROADMAP.md).

## Consequences

- The `YV_FORWARD` pipeline (`content_build.py`, the `build-content` CLI command,
  `frontend/content/forward.js`, the `<script>` tag, and `BlockContent`'s code
  block) is removed; a per-class source-link map replaces it.
- `Intuition` and `Theory` are the two prose pillars; both stay model-grounded.
- A new content channel is needed for Interpretation (per-type keying first).
- Recorded so the "don't reproduce PyTorch docs" boundary isn't quietly undone
  by future panel bloat.
