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
- **NEXT UP — hide scalar arithmetic op nodes.** Some `arith` op nodes are pure
  *shape arithmetic*, not data flow, but render as empty circles. Observations
  from block 10 (C2PSA), Attention expanded:
  - `matmul` / `matmul_1` are **already correct** — two genuine tensor inputs
    each (q·k = `[transpose, getitem_7] → [1,2,300,300]`; attn·v =
    `[getitem_8, transpose_1] → [1,2,64,300]`). No change needed.
  - `mul_1` (`matmul × 0.176` scale) is fine — one tensor input + a scalar
    *literal* (no spurious edge).
  - The confusing node is the bare **`mul` = `getitem_4 * getitem_5`** — both
    operands are `null`-shape scalars (computing `view` dims), `outShape: null`,
    no captured activation → an empty `arith` circle with no inputs/output.
  - Fix: **hide scalar op nodes** (call_function/arith whose instance shape is
    `null`), reusing the tensor-vs-scalar signal already used in the upstream
    walk (`instanceShapes(idx)[name]` not an array). Likely in `expand.jsx`
    `preprocessGraph` (alongside the existing `getitem` hide) or
    `classifySubkind`; confirm it doesn't hide tensor arith (`add`, `mul_1`).
- **Passthrough non-image-shaped outputs in the play-flow** (extends the shape-op
  passthrough). Nodes like `matmul`/`softmax` (`[1,2,300,300]`) and the attention
  strips (`[1,2,32,300]`) aren't reshapes, but their outputs aren't image-space,
  so a stretched heatmap in the flow misleads — give them the same prior-frame
  passthrough + friendly caption.
  - **Detection:** image-space ⟺ the output's last-two-dims aspect ratio is
    **proportional to the input image** (reference `YV_ACT.meta` image_w/h;
    equivalently any block's spatial grid — 20×15, 40×30, 80×60 all share it).
    *Any* resolution that's image-shaped qualifies (upscaling represents image
    information); non-spatial tensors (square 300×300, 32×300) passthrough. Use a
    small aspect tolerance for rounding.
  - Compose with the existing rule: passthrough if `isShapeOp(display)` **or**
    output not image-proportional. New caption for the value-op case (e.g.
    "attention scores — not an image").
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
  post-softmax attention row, reshaped to the attention grid and upsampled over the
  image (both heads selectable + mean). Design settled:
  - **Ship the attn tensor** (~313 KB uint8, both heads) and render client-side
    over the single input image — no baked frames. Capture is **eager**, folded
    into `build_assets`; `serve` computes it for uploaded images. Bundle
    precomputed JSON for a few sample images.
  - **Prototype status:** `codex/attention-viz-prototype` adds a standalone
    throwaway renderer at `frontend/attention-prototype.html`, fed by
    `scripts/export_attention_json.py` → `frontend/attention-prototype-data.js`.
    It supports query click/drag, head 0 / head 1 / mean, per-query vs global
    normalization, colormap/alpha controls, playback, and a visible state panel.
    Run with:
    `uv run python scripts/export_attention_json.py` then
    `uv run python -m http.server 8766 --directory frontend`.
  - **Prototype finding:** the grid is not always 20×20. It follows the
    letterboxed image aspect; the bundled sample produced 20×15
    (`shape=[2,300,300]`, 296 KB uint8 payload). Production copy/code should say
    "attention grid" rather than assuming a square grid.
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
