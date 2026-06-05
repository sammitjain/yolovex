# Attention tensor: capture mechanism and ship-payload contract

---
Status: accepted (UX choice superseded by ADR-0009; data contract still stands)
---

The Attention map (CONTEXT.md) needs the post-softmax attention matrix from
C2PSA[10]'s `attn` sub-module — a `[1, heads, N, N]` tensor (`N = gridH × gridW`)
that the normal activation-capture pipeline does not produce, because that
pipeline stores per-fx-node `mean / topK / stats`, not raw tensors. This ADR
records *how* the attention matrix is captured, *how* it is shipped to the
frontend, and *where* it lives in `window.YV_ACT`. The contract is hard to
reverse once the capture (`src/yolovex/`) and the renderer (`frontend/`) both
ship against it.

## Decision

### Capture — eager hook on `attn.forward`

`src/yolovex/attention_capture.py` (new) provides a small helper that
monkey-patches the `attn` submodule's `forward()` to stash the post-softmax
tensor, mirroring the validated prototype hook at
`scripts/viz_attention.py:53-87`:

```
attn = (q.transpose(-2, -1) @ k) * attn_module.scale
attn = attn.softmax(dim=-1)        # ← this is what we capture
captured["attn"] = attn.detach().cpu()
```

The helper is shared by `src/yolovex/build_assets.py` (default capture) and
`src/yolovex/serve.py` (upload-recompute), and by the standalone scripts
(`scripts/viz_attention.py`, `scripts/export_attention_json.py`) so the hook
lives in exactly one place.

**Why eager hook, not the fx interpreter:** the existing `_CapturingInterpreter`
records `mean / topK / stats` for every fx node; making it keep full raw tensors
for selected nodes would be a more invasive change to the capture core for a
single use case. The eager hook is one file, runs once, and matches the proven
prototype. If a second "raw tensor" use case appears, generalising the
interpreter at that point is the right move.

**Robustness:** the helper exposes a `restore()` callback used in a
`try/finally` so a forward-pass exception doesn't leave the model permanently
monkey-patched.

**Graceful absence:** if the model variant has no `attn` submodule at
`model.model.model[10].m[0].attn`, capture is skipped without erroring and the
frontend treats `YV_ACT.attention == null` as "feature unavailable".

### Quantisation — per-head uint8 + min/max

Reused verbatim from `scripts/export_attention_json.py:43-64`. Each head's
matrix is independently min-maxed and scaled to uint8 `[0, 255]`. Per-head
quantisation (not global) preserves dynamic range when one head's distribution
is much tighter than another's — common for self-attention. The renderer
dequantises in-browser via `value = min + (uint8 / 255) × (max − min)`.

Size: `heads × N × N` bytes raw (~180 KB for `2 × 300 × 300`), ~240 KB after
base64. Fits comfortably inside the existing `activations.js` payload.

### Payload location — top-level `window.YV_ACT.attention`

```js
window.YV_ACT.attention = {
  idx: 10,                              // C2PSA block index
  path: '0_PSABlock/attn/_op/softmax',  // the fx node this tensor came from
  heads: 2,
  gridH: 20, gridW: 15,                 // sample-specific; aspect-aware
  // ---- fields below match scripts/export_attention_json.py's payload exactly
  // so the prototype and production share one quantize/decode contract:
  dtype: 'uint8',
  shape: [heads, N, N],                 // N = gridH × gridW
  encoding: 'base64',
  quantization: 'per-head-linear',
  min: [min0, min1, ...],               // one entry per head
  max: [max0, max1, ...],
  data: '<base64 uint8 blob, heads × N × N, row-major>',
};
```

The fields from `dtype` down match `quantize_heads()` in
`src/yolovex/attention_capture.py` verbatim; the only additions for the in-app
context are the `idx` / `path` / `heads` / `gridH` / `gridW` envelope so the
frontend doesn't have to crack the spec to know where this tensor came from or
how to reshape it.

Sibling to `YV_ACT.nodes`, **not** nested inside `nodes['10'].sub[...]`. The
per-node `.sub` dict stores compact shape/stats per fx node; mixing a ~240 KB
raw-tensor blob into that struct would conflate two different kinds of data
(summary vs. raw) and would imply other nodes might also carry raw tensors when
they do not. A distinct top-level key makes the special status visible and
keeps lookup trivial (`if (YV_ACT.attention) { … }`).

`gridH × gridW` lives in the payload (not derived from the block's input shape)
because the renderer should not have to crack open the spec to know how to
reshape `dataB64`.

### Upload pathway

`serve.py`'s SSE upload already triggers a full `build_assets` rebuild for each
uploaded image. Because the eager hook lives inside `build_assets`, the upload
path picks up the new payload for free — no separate SSE event, no parallel
code path.

## Consequences

- The frontend's only entry point for attention data is `window.YV_ACT.attention`.
  Anything reading the softmax fx node's `mean / stats` from `nodes['10'].sub`
  stays correct for other purposes (channel brochure, IO strip) — those keep
  their existing summary representation.
- The contract is keyed on **what the model produces**, not on what the UI happens
  to render. If a future view wants the value tensor `V` or the un-softmaxed
  scores, that's a *separate* payload (e.g. `YV_ACT.attentionScores`), not an
  overload of this one.
- Per-head uint8 quantisation is good enough for visual rendering but does lose
  precision. Anything that needs the exact float values (e.g. an offline
  numerical analysis) should use `scripts/export_attention_json.py` directly,
  which produces the same payload format (and could trivially be switched to
  float32 by changing one helper).
