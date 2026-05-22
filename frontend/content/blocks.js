// Hand-curated explainer text — pulled from the YOLOVEX research doc and
// distilled for side-panel display. Keyed by class name (matches the
// `type` field on outer blocks AND the `target_class` resolved by the
// fx capture for sub-nodes).
//
// Schema per entry (see ADR-0003 for the panel content model):
//   title      — panel header title (was app.jsx TYPE_COPY)
//   blurb      — one-paragraph header description under the title
//   tagline    — single line, shows under the title in the explainer section
//   intuition  — 2–3 sentences (markdown), the friendly hook
//   technical  — Theory: this model's mechanics/params (markdown), in context;
//                never a free-floating textbook definition
//   shape      — optional shape rule of thumb (one line)
//   yolo26     — optional callout for YOLO26-specific quirks
//   refs       — optional citations / source links { label, url }
//
// Position-specific copy (notes true only at one block index, e.g. the
// attn=True C3k2 at [22]) lives in YV_CONTENT_OVERRIDES below, keyed by
// position — NOT in the per-type entries here.
//
// All strings are GFM markdown. Code spans are styled inline; fenced blocks
// render with monospace. Keep it tight — the panel is bounded.

window.YV_CONTENT = {

  // =========================================================================
  // Outer blocks
  // =========================================================================

  Conv: {
    title: "Conv block — Conv2d → BN → SiLU",
    blurb: "The basic building block. A learnable Conv2d does feature extraction, BatchNorm normalizes the result, SiLU applies a smooth nonlinearity. Stride-2 versions also halve the spatial size.",
    tagline: "Conv2d → BatchNorm2d → SiLU (the network's main building block).",
    intuition:
      "Every `Conv` learns a local pattern (the kernel), normalizes its response so training stays stable, and squashes negatives through a smooth gate. Stacking many of them is how the network builds a hierarchy: edges → textures → parts → objects.",
    technical:
      "A fused triple: `nn.Conv2d` (no bias — the next BN absorbs it), `nn.BatchNorm2d`, and `nn.SiLU`. In the YOLO26 backbone every *downsample* step is a `Conv(k=3, s=2)`; everywhere else a 1×1 `Conv` is a pure pointwise channel-mixer (e.g. `cv1`/`cv2` inside C3k2 / C2PSA).",
    shape:
      "`(k=3, s=1)` preserves H,W and mixes channels · `(k=3, s=2)` halves H,W and mixes · `(k=1)` is a pointwise channel mixer.",
  },

  C3k2: {
    title: "C3k2 block — CSP feature mixer",
    blurb: "Splits the input into two halves, runs a Bottleneck (or nested C3k) on one half, then concats both halves back with the bottleneck output before a 1×1 projection. YOLO26's main feature-mixing unit at every backbone resolution.",
    tagline: "CSP-style feature mixer (split → bottlenecks → concat → project).",
    intuition:
      "Split the channels in two parallel streams. Keep one stream as-is (the gradient highway). Pour the other through a stack of bottleneck transformations. Then weave the streams back together. Half the work, most of the signal.",
    technical:
      "Inherits the CSP forward from `C2f`: `cv1` widens, `.chunk(2, dim=1)` halves channels, the second half feeds through `self.m = ModuleList(Bottleneck or nested C3k, repeat n)`, all branches are `cat`'d, `cv2` projects out. The four knobs are `c1`, `c2`, `n` (depth, scaled by `depth_multiple`), and `c3k` (Bottleneck vs nested C3k inner). The expansion `e` (default 0.5) sets the split width.",
  },

  SPPF: {
    title: "SPPF — spatial pyramid pooling (fast)",
    blurb: "Stacks three 5×5 max-pools so a single forward gives features at 4 receptive-field scales, concats them, and projects with a 1×1. Cheap way to give the deepest backbone layer a \"view\" at multiple object sizes.",
    tagline: "Spatial Pyramid Pooling — Fast. Multi-scale receptive field with one kernel.",
    intuition:
      "Every cell should know not just its 3×3 neighbourhood, but also its 5×5, 9×9 and 13×13 neighbourhood — so a prediction can use both local and broad context. SPPF gets all three by running **one** `MaxPool(k=5)` three times in sequence (k=5 twice = k=9 receptive field, three times = k=13).",
    technical:
      "`cv1` halves channels (`act=False` — SiLU would be wasted under the max-pool). Three sequential `MaxPool2d(k=5, s=1, p=2)` produce receptive fields of 5, 9, 13. All four maps (`cv1(x)` + 3 pools) are concatenated along channels, `cv2` projects back, and YOLO26 adds a residual `+ x` when `shortcut=True` and channels match.",
    yolo26:
      "**SPPF shortcut** (one of the two main YOLO26 changes vs YOLO11). The new `+ x` lets gradients short-circuit straight to the SPPF input and lets the network fall back to un-pooled features when the pyramid isn't useful for a sample.",
  },

  C2PSA: {
    title: "C2PSA — CSP + position-sensitive attention",
    blurb: "Like C3k2 but with a self-attention block on the parallel branch. Adds long-range context on top of the local convolutional features at the deepest scale.",
    tagline: "Cross-Stage Partial with Position-Sensitive Self-Attention.",
    intuition:
      "The network's *global-context* module. After the backbone extracts strong local features, C2PSA lets every position attend to every other position once — so the network can reason about object-object relationships and long-range structure before Detect emits boxes.",
    technical:
      "CSP-shaped: `cv1` widens, `.split` halves channels, only the second half (`b`) goes through the attention stack `self.m = Sequential(PSABlock × n)`. The two halves are `cat`'d and `cv2` projects back. Lives at block 10 — the deepest, smallest, most semantic map (20×20 at 640 input), which is the cheapest place to do O(N²) attention. yolo26n has `n=1` (one PSABlock) after `depth_multiple=0.50`.",
  },

  Upsample: {
    title: "Upsample — 2× nearest neighbor",
    blurb: "Doubles spatial dimensions by repeating each pixel in a 2×2 block. No learnable params. The neck uses this to bring deep coarse features back up to higher-resolution scales for small-object detection.",
    tagline: "Nearest-neighbour 2× upsampler. Zero parameters.",
    intuition:
      "Each input pixel becomes a 2×2 block of identical pixels in the output. Crude, but the *information* is preserved exactly — downstream convolutions smooth and refine it.",
    technical:
      "`nn.Upsample(scale_factor=2, mode='nearest')`. `(B, C, H, W) → (B, C, 2H, 2W)`. Used in the FPN-up path so a deep feature (`P5`, 20×20) can be made the same shape as a shallower one (`P4`, 40×40) before they're `Concat`'d. Nearest-neighbour over transposed-conv: faster, no learned params, ONNX/TensorRT-friendly.",
  },

  Concat: {
    title: "Concat — stack channels",
    blurb: "Joins multiple tensors along the channel dimension. Output channel count = sum of inputs. Used inside C3k2/SPPF to merge parallel branches, and across the neck to fuse skip connections with upsampled features.",
    tagline: "Channel-wise glue. The fusion point of the neck.",
    intuition:
      "Two transparent acetate sheets laid on top of each other. The next `C3k2` learns the optimal blend. Every FPN/PAN fusion in the neck is a `Concat`: upsampled-deep + same-resolution-shallow → the next block decides the mix.",
    technical:
      "`torch.cat([...], dim=1)`. No learnable parameters, no FLOPs beyond a memory copy. Two `(B, c_a, H, W)` and `(B, c_b, H, W)` tensors produce `(B, c_a + c_b, H, W)`.",
  },

  Detect: {
    title: "Detect head — anchor-free, NMS-free",
    blurb: "Runs three parallel conv heads (one per scale) and emits boxes + class scores per anchor cell. Final detections come from a top-K filter on the one-to-one branch (NMS-free).",
    tagline: "Per-scale class scores + box coordinates. End-to-end, NMS-free.",
    intuition:
      "The point where the network finally *says something* about objects. Consumes three neck feature maps (`P3` 80×80, `P4` 40×40, `P5` 20×20) and produces per-cell class scores + bounding boxes at each scale. Almost all of YOLO26's novelty lives here.",
    technical:
      "Each scale has its own decoupled stack: `cv3` (class branch) → `(B, nc, H, W)` after a 1×1 conv; `cv2` (box branch) → `(B, 4·reg_max, H, W)`. Class probabilities are *independent* (sigmoid per class, no softmax), because one object can legitimately match multiple labels.",
    yolo26:
      "Three big changes:\n\n- **DFL removed** (`reg_max=1`): each edge is a single scalar, not a distribution over 16 bins — fewer FLOPs, cleaner export.\n- **Dual-head training** (one-to-many + one-to-one): the one-to-many head gives dense gradient signal during training; the one-to-one head is what runs at inference (no NMS needed).\n- **Top-K inference**: instead of NMS, the head emits the K=300 highest-confidence predictions across scales and classes. Output is `(B, 300, 6)` = `(x1, y1, x2, y2, score, class_id)`. Truly end-to-end.",
  },

  // =========================================================================
  // Sub-component classes — surfaced when shift+click peels a block open
  // =========================================================================

  Conv2d: {
    title: "Conv2d — learnable filter bank",
    blurb: "Slides a small kernel across the input and produces one output channel per filter. This is the ONLY layer in the Conv block that carries learnable weights — it does the actual feature extraction. Stride > 1 here also means spatial downsampling.",
    tagline: "Learnable spatial filter sliding over the input.",
    intuition:
      "Each kernel is a tiny pattern detector — \"horizontal edge?\", \"red-brown blob?\" — applied at every location. The output map is *the response of this detector everywhere in the image*.",
    technical:
      "Three knobs that matter:\n\n- **kernel size `k`**: `k=3` is the workhorse; `k=1` is a per-pixel channel-mixer.\n- **stride `s`**: `s=1` preserves H,W; `s=2` halves them.\n- **padding `p`**: Ultralytics uses `autopad` so `k=3, s=1` outputs match input H,W exactly.\n\n`bias=False` everywhere — the following BatchNorm's shift makes the conv bias redundant.",
  },

  BatchNorm2d: {
    title: "BatchNorm — channel-wise normalization",
    blurb: "Re-centers and re-scales each channel using running mean/variance learned during training. Keeps activations in a stable range so the next nonlinearity sees predictable inputs.",
    tagline: "Per-channel zero-mean / unit-variance, plus learned scale + shift.",
    intuition:
      "A thermostat that constantly readjusts each channel's volume knob, so no single feature dominates and drowns the others out. Lets you train with higher learning rates without diverging.",
    technical:
      "Normalises each channel of `(B, C, H, W)` across the `(B, H, W)` dims, then applies learned `γ` (scale) and `β` (shift) per channel. Tracks running mean/var during training; freezes them at inference. **`Conv` + `BN` can be fused into a single Conv at inference** — that's what `model.fuse()` does, and why yolo26n's published 2.4M params is the *after-fusion* count.",
    refs: [{ label: "Ioffe & Szegedy, 2015", url: "https://arxiv.org/abs/1502.03167" }],
  },

  SiLU: {
    title: "SiLU — smooth nonlinearity",
    blurb: "Computes x · sigmoid(x). For most positive activations SiLU is nearly x (so this thumbnail will look almost identical to BN output above), but it gently suppresses strong negative values.",
    tagline: "Smooth, non-monotonic activation: `x · sigmoid(x)`.",
    intuition:
      "ReLU is a strict bouncer — negative? out. SiLU is the polite bouncer that lets slightly-negative values through with a discount, and gives positive values a small smooth boost near the origin.",
    technical:
      "Also called **Swish**. Smooth and differentiable everywhere (better gradient flow), with a small negative dip near `x ≈ -1.28` that empirically improves expressiveness in deep CNNs. No \"dying ReLU\" problem — units can't get stuck at zero. Ultralytics has used SiLU as the default Conv activation since YOLOv5.",
    refs: [{ label: "Ramachandran et al., 2017 (Swish)", url: "https://arxiv.org/abs/1710.05941" }],
  },

  MaxPool2d: {
    title: "MaxPool — local maximum",
    blurb: "Inside SPPF, a 5×5 stride-1 max pool with padding 2. SPPF chains it three times so the n-th call has covered an effective receptive field of (4n+1)×(4n+1) — a \"pyramid of receptive fields\" without changing spatial size.",
    tagline: "Ask each region for its loudest voice.",
    intuition:
      "Slide a window across the feature map and emit the maximum value in each window. Cheap, parameterless, monotonic.",
    technical:
      "SPPF uses `MaxPool2d(k=5, s=1, p=2)` three times in sequence. The receptive field after `n` pools is `1 + n·(k-1)` — so after 1, 2, 3 pools, each cell summarises a 5×5, 9×9, 13×13 neighbourhood. Combined with the un-pooled `cv1(x)` (RF = 1), this is exactly the SPP pyramid, achieved with shared computation.",
  },

  DWConv: {
    tagline: "Depthwise convolution — one filter per channel.",
    intuition:
      "Normal Conv2d mixes channels at every location. Depthwise Conv2d keeps channels *separate* — each input channel gets its own filter. Cheap, and often used as a position embedding.",
    technical:
      "Equivalent to `nn.Conv2d(c, c, k, groups=c)`. Inside `Attention.pe`, a depthwise 3×3 conv is applied to the values `V` and added to the attention output — this is the **position-sensitive** twist that re-injects local spatial structure after self-attention's permutation-invariant mix.",
  },

  Bottleneck: {
    title: "Bottleneck — residual mini-block",
    blurb: "Two 3×3 convs sandwiching the channel count, with an additive residual skip. Lets each C3k2 add a couple of layers of nonlinear refinement on the parallel branch without inflating param count.",
    tagline: "Classic ResNet 3×3 → 3×3 with optional residual.",
    intuition:
      "Two stacked 3×3 convs with a skip connection (when channels match). The 3×3s do the real spatial mixing; the residual keeps gradients alive when the network is deep.",
    technical:
      "`forward(x) = x + cv2(cv1(x))` if `shortcut and c1 == c2`, else `cv2(cv1(x))`. The default Bottleneck inside C3k2 is built with `cv1=Conv(c1, c_, k=3)` and `cv2=Conv(c_, c2, k=3)` where `c_ = int(c2 * e)`.",
  },

  C3k: {
    title: "C3k — small CSP sub-block",
    blurb: "A miniature CSP module (split → series of bottlenecks → concat → 1×1) used inside the deeper C3k2 layers.",
    tagline: "Tiny nested CSP block — \"a CSP inside a CSP.\"",
    intuition:
      "When C3k2 is built with `c3k=True`, its inner repeated module is a `C3k` instead of a plain Bottleneck. That nested CSP structure gives deeper blocks more representational capacity per channel.",
    technical:
      "Inherits from `C3`. Two parallel 1×1 paths (`cv1` and `cv2`), one of which goes through a Sequential of Bottlenecks; the two are concatenated and projected by `cv3`. So a `C3k2(c3k=True, n=k)` is *k little CSPs inside a CSP*.",
  },

  PSABlock: {
    title: "PSABlock — position-sensitive self-attention",
    blurb: "Runs self-attention (Q/K/V) followed by a small feed-forward. Lets the network propagate information across distant locations — useful at the deepest backbone scale where a single token already covers a big receptive field.",
    tagline: "Transformer-style Attention → FFN, with residuals — on 2D feature maps.",
    intuition:
      "The Transformer recipe, adapted for `(B, C, H, W)` tensors instead of token sequences. Attention lets each position look anywhere; the FFN does per-token channel mixing; residuals keep gradients flowing.",
    technical:
      "`forward(x) = x + attn(x); x = x + ffn(x); return x` (with `shortcut=True`, the default). The attention is the `Attention` module (qkv + scaled dot-product + position-sensitive `pe`); the FFN is `Conv(c, 2c, k=1) → Conv(2c, c, k=1, act=False)` — the second conv has no activation so the FFN contribution is a clean linear add to the residual.",
  },

  Attention: {
    tagline: "Single-shot self-attention with a depthwise position bias.",
    intuition:
      "Each position is allowed to look around the whole feature map and pull information from wherever it finds useful. A position over a person's foot might attend strongly to the head to confirm \"this is a person\" before refining its box.",
    technical:
      "A single 1×1 conv `qkv` produces Q, K, V in one go and they're split along channels. `attn = softmax((Qᵀ K) · scale)` is an `N×N` matrix where `N = H·W` — each row says how much this position should weight every other when reading values. A depthwise 3×3 conv `pe` is applied to V and added back to the attention output: the **\"position-sensitive\"** twist that re-injects locality (pure attention is permutation-invariant — it forgets where tokens are). A final 1×1 `proj` mixes heads back.",
  },

  Upsample_torch: {
    // Defensive duplicate in case the class is resolved as plain torch.nn.Upsample
    // The outer Upsample entry above already covers it; this is just a safety net.
    tagline: "See Upsample.",
    intuition: "See `Upsample`.",
    technical: "Same as the outer `Upsample` entry.",
  },

  // ===========================================================================
  // Functional-op scaffolds — SCAFFOLD/PLACEHOLDER, not researched copy.
  // Routed to by op `target` when a sub-node has no module class (cat is
  // instead routed to Concat; see app.jsx). Replace with researched copy
  // (see ROADMAP.md).
  // ===========================================================================

  add: {
    title: "add — residual / element-wise sum",
    tagline: "Element-wise addition (placeholder).",
    intuition:
      "_Scaffold._ Adds two tensors element-wise — typically a residual connection that lets the input bypass a transformation and keeps gradients flowing.",
  },

  split: {
    title: "split — channel split",
    tagline: "Splits a tensor into chunks along a dimension (placeholder).",
    intuition:
      "_Scaffold._ Divides the tensor into parts (usually halving channels) so a CSP-style block can send one part down a transform branch and keep the other as a shortcut.",
  },

  chunk: {
    title: "chunk — channel split",
    tagline: "`tensor.chunk(n, dim)` (placeholder).",
    intuition:
      "_Scaffold._ Like `split` — breaks the tensor into `n` equal parts along a dimension; C3k2 uses `.chunk(2, dim=1)` to halve channels.",
  },

  getitem: {
    title: "getitem — index / slice",
    tagline: "Selects one element of a tuple/list output (placeholder).",
    intuition:
      "_Scaffold._ Picks a single output out of a multi-output op (e.g. one half of a `chunk`/`split` result) so it can feed downstream.",
  },
};

// Position-specific overrides — copy that is true only at one position, keyed by
// position: `String(idx)` for an L1 block, or `` `${idx}/${pathKey}` `` for a
// sub-node. Merged over the per-type entry (override wins per field) in the
// panel, so e.g. the attn=True C3k2 at block 22 shows this YOLO26 note while
// every other C3k2 does not.
window.YV_CONTENT_OVERRIDES = {

  "22": {
    yolo26:
      "**Block 22 (final C3k2 on the P5 path) is special.** The YAML enables a new `attn=True` flag so its inner repeated module is a `PSABlock` instead of a Bottleneck. Cheap attention right before Detect, exactly where the spatial size is smallest (20×20) but the semantics are richest.",
  },

};
