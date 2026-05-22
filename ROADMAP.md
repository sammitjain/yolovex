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
  `BlockContent` (`app.jsx`). Working but **untracked in git**. To do:
  - **Drop the `forward()` pipeline** — delete `content_build.py`, the
    `build-content` CLI command, `frontend/content/forward.js`, its `<script>`
    tag, and `BlockContent`'s code block (ADR-0003: canvas already shows the
    wiring; link to source instead).
  - **Add a per-class source-link map** — Ultralytics for the YOLO blocks
    (`block.py` composites, `conv.py` for Conv/DWConv/Concat, `head.py` for
    Detect), PyTorch docs for stdlib leaves.
  - **Widen `blocks.js` coverage** to every block/sub-node class, consistently.
  - **Replace op scaffolds with researched copy** — `add`/`split`/`chunk`/
    `getitem` in `blocks.js` are placeholder `_Scaffold._` text; author real
    descriptions. Also add `title`/`blurb` for `DWConv`/`Attention`/
    `Upsample_torch` (no `TYPE_COPY` ever existed for them, so they fall back to
    the bare class name in the header).
  - **Author per-position overrides** as needed in `YV_CONTENT_OVERRIDES`
    (keyed by `idx` or `idx/pathKey`) — only block 22's C3k2 note exists today.
  - Commit once stable.

## Next

- **Content audit — every block / node / op (ADR-0003).** Full pass over
  `blocks.js`: confirm each entry's copy is present and *appropriate* for that
  class (add what's missing, fix what's off), now that routing lands copy on the
  correct node. Covers L1 blocks, sub-node module classes, and the op scaffolds.
  Needs a final human review pass (the descriptions are research-grade).
- **Activation capture for shape/op nodes.** Some nodes (e.g. `.chunk()`,
  `split`, `getitem`) show "No 4-D tensor captured" and have no channel brochure,
  even though they do produce a viewable tensor. Suspected cause: the activation-
  gathering pass (Python capture → `activations.js` / `YV_ACT`) isn't wired for
  these op outputs. Investigate the capture path and emit their tensors so the
  brochure renders.
- **Interpretation layer (ADR-0003, layer 3 — not built).** Hand-authored,
  image-robust "what to look for in this activation" note. **L1 canvas blocks
  first**, sub-nodes/leaves incrementally (second priority). Per-type keying to
  start. Depends on / fed by the attention-viz work for the attention blocks.
- **Attention visualization.** `scripts/viz_attention.py` animates C2PSA
  self-attention per query pixel (standalone, exploratory). Decide whether it
  graduates into the frontend (e.g. an in-panel attention view) or stays a
  research/asset script — and what artifact it produces. Doubles as the source
  for the C2PSA / Attention / PSABlock Interpretation notes above.

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

- ELK adopted as in-Region layout engine; placement + routing kept inside ELK,
  no post-process (ADR-0001, ADR-0006).
- In-Region edges rendered as ELK splines with flattened port tails (ADR-0007).
