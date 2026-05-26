# yolovex — Roadmap

The running list of **intended / next work**. This is the home for work items —
ADRs record decisions *already made*, `CONTEXT.md` is a glossary. When a roadmap
item forces a real, hard-to-reverse trade-off, spin that decision off into an
ADR and link it here.

Keep entries short. Move an item to **Done** (with the resolving commit/ADR)
when it lands; prune Done periodically into git history.

## Next

- **Side-panel content audit (ADR-0003).** Full pass over
  `frontend/content/blocks.js`: confirm copy is present and appropriate for L1
  Blocks, Sub-node module classes, and op nodes. Specific gaps:
  - Add a per-class source-link map (Ultralytics for YOLO blocks; PyTorch docs
    for stdlib Leaves). `refs` exists on only 2 entries today.
  - Widen coverage to every block/sub-node class.
  - Replace op scaffolds (`add`/`split`/`chunk`/`getitem`) with researched copy.
  - Add `title`/`blurb` for `DWConv`/`Attention`/`Upsample_torch`.
  - Rewrite mechanical play-flow captions and panel titles for learners.
  - Add per-position overrides in `YV_CONTENT_OVERRIDES` where useful.
- **Attention visualization — production integration.** Promote the validated
  prototype into the app: C2PSA/Attention view, query patch click/drag, head
  selector, mean view, colormap/alpha controls, and upload support.
  - Keep `scripts/viz_attention.py` as the standalone GIF/MP4 generator.
  - Fold eager attention capture into `build_assets`; `serve` should compute it
    for uploaded images.
  - Ship the attn tensor (~300 KB uint8 for the sample) and render client-side
    over the input image; bundle precomputed JSON for sample images.
  - Record the ship-tensor + client-render data contract in an ADR before
    productionizing it.
  - Use `docs/attention-visualization.md` and the prototype finding that grids
    follow the letterboxed image aspect (sample: 20×15, not always 20×20).
- **Interpretation layer (ADR-0003, layer 3 — not built).** Hand-authored,
  image-robust "what to look for in this activation" note. **L1 canvas blocks
  first**, sub-nodes/leaves incrementally (second priority). Per-type keying to
  start. Depends on / fed by the attention-viz work for the attention blocks.

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

- **ELK layout is the primary page; old renderer removed.** `/` now serves the
  ELK explorer (`index-elk.html` → `index.html`; old `index.html` and
  `graph.jsx` deleted, history is the archive). `expand.jsx` was split: its
  shared semantic front-half (`_graphSem`, palettes, sizing) moved to the new
  `graph-sem.jsx`; the dead geometry back-half (legacy `buildExpansion`,
  `detectStaircases`, skip-lane/staircase machinery superseded by ADR-0006) went
  with the file. `layout.jsx`/`arch.jsx` were *not* dead — they're the shared L1
  cross-Block engine. Docs refreshed (`serve.py`, `frontend/README.md`,
  `docs/preview-and-verification.md`).
- **Attention visualization prototype.** Standalone prototype merged on
  `codex/attention-viz-prototype` and into local `main` via d3585f8:
  `scripts/export_attention_json.py`, `scripts/serve_attention_prototype.py`,
  `frontend/attention-prototype.html`, sample
  `frontend/attention-prototype-data.js`, and learner notes in
  `docs/attention-visualization.md`. Validated query click/drag, heads + mean,
  per-query/global normalization, colormap/alpha controls, playback, upload, and
  the aspect-aware attention grid (sample is 20×15).
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
