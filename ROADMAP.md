# yolovex — Roadmap

The running list of **intended / next work**. This is the home for work items —
ADRs record decisions *already made*, `CONTEXT.md` is a glossary. When a roadmap
item forces a real, hard-to-reverse trade-off, spin that decision off into an
ADR and link it here.

Keep entries short. Move an item to **Done** (with the resolving commit/ADR)
when it lands; prune Done periodically into git history.

## In flight

- **Side-panel content (ADR-0003).** Hand-curated `frontend/content/blocks.js`
  (tagline / intuition / Theory(`technical`) / yolo26 / shape) merged in
  `BlockContent` (`app.jsx`); now tracked in git. To do:
  - **Add a per-class source-link map** — Ultralytics for the YOLO blocks
    (`block.py` composites, `conv.py` for Conv/DWConv/Concat, `head.py` for
    Detect), PyTorch docs for stdlib leaves. (`refs` exists on only 2 entries
    today.)
  - **Widen `blocks.js` coverage** to every block/sub-node class, consistently.
  - **Replace op scaffolds with researched copy** — `add`/`split`/`chunk`/
    `getitem` in `blocks.js` are still placeholder `_Scaffold._` text; author
    real descriptions. Also add `title`/`blurb` for `DWConv`/`Attention`/
    `Upsample_torch` (no `TYPE_COPY` ever existed for them, so they fall back to
    the bare class name in the header).
  - **Author per-position overrides** as needed in `YV_CONTENT_OVERRIDES`
    (keyed by `idx` or `idx/pathKey`) — only block 22's C3k2 note exists today.

## Next

- **Content audit — every block / node / op (ADR-0003).** Full pass over
  `blocks.js`: confirm each entry's copy is present and *appropriate* for that
  class (add what's missing, fix what's off), now that routing lands copy on the
  correct node. Covers L1 blocks, sub-node module classes, and the op scaffolds.
  Needs a final human review pass (the descriptions are research-grade). Also:
  **friendlier learner-facing strings** — the play-flow overlay captions and
  side-panel titles read mechanically / code-flavoured today; rewrite for a
  learner (drop fx-node nuance from titles).
- **Make the ELK layout the primary page; archive the old layout.**
  - `serve.py` mounts `frontend/` at `/` with `html=True` → `/` serves
    `index.html`. Switch primary to the ELK page (rename `index-elk.html` →
    `index.html` after archiving the current one, or adjust the mount); update
    `serve.py`'s docstring + `frontend/README.md`.
  - ELK-only: `graph-elk.jsx`, `expand-elk.jsx`, `vendor/elk.bundled.js`.
    Old-renderer-only: `graph.jsx` (archive/remove). Shared: `arch.jsx`,
    `layout.jsx`, `expand.jsx`, `app.jsx`.
  - Needs a **dead-code audit**: parts of `expand.jsx`/`layout.jsx`/`arch.jsx`
    used only by the retired `graph.jsx` path (e.g. `detectStaircases` /
    skip-lane machinery superseded by ADR-0005/0006) can go. Minor refactor; do
    the removal carefully.
- **Interpretation layer (ADR-0003, layer 3 — not built).** Hand-authored,
  image-robust "what to look for in this activation" note. **L1 canvas blocks
  first**, sub-nodes/leaves incrementally (second priority). Per-type keying to
  start. Depends on / fed by the attention-viz work for the attention blocks.
- **Attention visualization.** Graduate `scripts/viz_attention.py` into an
  interactive in-frontend attention view: pick/drag a query patch → its
  post-softmax attention row, reshaped to the 20×20 grid and upsampled over the
  image (both heads selectable + mean). Design settled:
  - **Ship the attn tensor** (~313 KB uint8, both heads) and render client-side
    over the single input image — no baked frames. Capture is **eager**, folded
    into `build_assets`; `serve` computes it for uploaded images. Bundle
    precomputed JSON for a few sample images.
  - **Prototype the look first** in a standalone HTML fed by a JSON dump from the
    existing `capture_attention()`; iterate on colormap / alpha / interpolation /
    interaction, then port the renderer into the app.
  - The GIF/MP4 script stays a standalone asset generator. Feeds the
    C2PSA / Attention / PSABlock Interpretation notes above.
  - **Open (defer to ADR after prototype):** the ship-tensor + client-render data
    contract is a real, hard-to-reverse trade-off; record it once the prototype
    validates the approach.

## Later

- **Richer IO-strip rendering.** Today it's a plain "Input → Output" row
  (`in1 + in2 → out` for Merges). Open to a better visual for the input→output
  transform. Low priority.
- **L1 edge refinement.** Specific cross-Block edges (e.g. the residual into
  SPPF's `add`) are a `layout.jsx` concern, not an ELK knob (see ADR-0006/0007).
  Tune on feedback rather than pre-emptively.
- **Public hosting (ADR-0004).** Decision made (HF Docker Space) but not yet
  executed.

## Done (recent)

- **IO strip surfaces the real split slice + scalar arith operands (ADR-0003).**
  Two attention-block confusions fixed. (1) `subUpstreamSources` (`app.jsx`) now
  stops the upstream walk at a *captured* `getitem` (a split's per-piece slice)
  instead of always passing through it — so `matmul_1`'s input reads as the V
  slice `[1,2,64,300]`, not the pre-split `view` `[1,2,128,300]` (the dims now
  multiply). Same rule makes a C3k2 `cat` show its three real `16`-ch slices
  rather than dedup'ing them to the pre-chunk parent — confirmed acceptable.
  Safe because the IO strip renders no fx node names, only thumbnails/shapes.
  (2) A binary arith op against a constant (one tensor arg + one numeric literal)
  now carries the literal through `aggregateWithExpansions` → `expand-elk.jsx` →
  `graph-elk.jsx`, which renders it under the node (e.g. attention's score
  scaling shows `× 0.177` = 1/√d_k) so the one-input `×` circle isn't a mystery.
  Also fixed the arith glyph table (was keyed on `mul`/`add` but labels are
  `fn:mul`/`fn:add`, so circles showed text not `×`/`+`). Verified live on
  `index-elk.html`.
- **Play-flow passthrough for non-image-shaped outputs (ADR-0003).** The overlay
  now stretches an activation to the image frame only when it's *image-shaped* —
  its last-two-dims aspect is proportional to the input image (`isImageShaped`
  in `app.jsx`, referencing `YV_ACT.meta` image_w/h, small tolerance for grid
  rounding). Captured-but-non-image tensors (attention scores `[1,2,300,300]`,
  value strips `[1,2,*,300]`) now passthrough the prior frame with a "not an
  image" caption, composing with the existing shape-op passthrough. Verified
  live on `index-elk.html` (conv grids stretch; matmul/softmax passthrough).
  *Scalar arith op nodes* (the bare `mul = getitem_4 * getitem_5`, `outShape:
  null`) were the sibling concern here — already handled: `preprocessGraph`'s
  existing `shapes[name] === null` drop removes every null-shape node, so they
  never reach the canvas as empty circles (confirmed for block 10 / Attention).
- **Activation view for op nodes (ADR-0003).** Split ops (`.split()`/`.chunk()`)
  → per-output brochures (out 0/1/2 tabs, sourced from the already-captured
  `getitem` children, sizes read off each piece's shape); shape ops →
  shape-transformation IO card + passthrough-with-caption in the play-flow;
  channel brochure preview *and* IO-strip tiles now render at the tensor's true
  H×W via `fitBox` (`app.jsx`); `subUpstreamSources` is tensor-aware (drops
  `shape==null` scalar args — fixed the reshape "2 inputs" bug); brochure-thumb
  `min-height` keeps the channel label visible on wide-short maps. Frontend-only;
  verified on `index-elk.html`.
- Side-panel `forward()` pipeline dropped (`content_build.py`, `build-content`
  CLI, `YV_FORWARD`, `forward.js` removed); copy routing fixed; content moved to
  data in `blocks.js`, now tracked — commit d3d8693, ADR-0003.
- ELK adopted as in-Region layout engine; placement + routing kept inside ELK,
  no post-process (ADR-0001, ADR-0006).
- In-Region edges rendered as ELK splines with flattened port tails (ADR-0007).
