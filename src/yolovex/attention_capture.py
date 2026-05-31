"""Capture the post-softmax attention tensor from C2PSA's `attn` sub-module.

The eager hook here is shared by:
- `src/yolovex/build_assets.py` — folds attention into the default activations
  payload (per ADR-0008).
- `src/yolovex/serve.py` — reuses build_assets, so the upload path picks it up
  for free.
- `scripts/viz_attention.py` and `scripts/export_attention_json.py` — keep
  importing from here rather than duplicating the hook.

See ADR-0008 for the ship-payload contract.
"""

from __future__ import annotations

import base64
from typing import Any, Callable

import numpy as np


SOFTMAX_FX_PATH = "0_PSABlock/attn/_op/softmax"


def find_attention_module(blocks) -> tuple[int, Any] | tuple[None, None]:
    """Locate the C2PSA `attn` sub-module. Returns (idx, module) or (None, None)
    if this model variant doesn't have one — the graceful-absence path."""
    for idx, b in enumerate(blocks):
        if type(b).__name__ != "C2PSA":
            continue
        try:
            return idx, b.m[0].attn
        except (AttributeError, IndexError):
            continue
    return None, None


def attach_attn_hook(attn_module) -> tuple[dict, Callable[[], None]]:
    """Monkey-patch attn_module.forward to stash the post-softmax attn tensor.

    Returns (captured, restore). `captured` will be populated with
    `{'attn': Tensor[B, heads, N, N], 'H': int, 'W': int}` after a forward pass.
    Always call `restore()` in a finally — leaving the patch in place would
    leak across subsequent forwards.
    """
    captured: dict = {}
    original_forward = attn_module.forward

    def patched_forward(x):
        B, C, H, W = x.shape
        N = H * W
        qkv = attn_module.qkv(x)
        q, k, v = qkv.view(
            B, attn_module.num_heads,
            attn_module.key_dim * 2 + attn_module.head_dim, N
        ).split(
            [attn_module.key_dim, attn_module.key_dim, attn_module.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * attn_module.scale
        attn = attn.softmax(dim=-1)
        captured["attn"] = attn.detach().cpu()
        captured["H"] = H
        captured["W"] = W
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + attn_module.pe(
            v.reshape(B, C, H, W)
        )
        return attn_module.proj(out)

    attn_module.forward = patched_forward

    def restore() -> None:
        attn_module.forward = original_forward

    return captured, restore


def quantize_heads(attn: np.ndarray) -> dict[str, Any]:
    """Quantize [heads, N, N] float attention to a per-head uint8 blob.

    Per-head linear quantization preserves dynamic range when one head's
    distribution is much tighter than another's (common for self-attention).
    Output is the canonical shape consumed by both the prototype JSON payload
    and the YV_ACT.attention payload (ADR-0008).
    """
    if attn.ndim != 3:
        raise ValueError(f"expected [heads,N,N], got shape {attn.shape}")

    heads = attn.shape[0]
    flat = attn.reshape(heads, -1).astype(np.float32)
    mins = flat.min(axis=1)
    maxs = flat.max(axis=1)
    spans = np.maximum(maxs - mins, 1e-12)
    q = np.round((flat - mins[:, None]) / spans[:, None] * 255.0)
    q = np.clip(q, 0, 255).astype(np.uint8).reshape(attn.shape)

    return {
        "dtype": "uint8",
        "shape": list(q.shape),
        "encoding": "base64",
        "quantization": "per-head-linear",
        "min": [float(v) for v in mins],
        "max": [float(v) for v in maxs],
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def build_attention_payload(
    captured: dict,
    *,
    idx: int,
    path: str = SOFTMAX_FX_PATH,
) -> dict[str, Any] | None:
    """Wrap a captured attention tensor in the YV_ACT.attention payload shape.

    Returns None if the hook never fired (captured dict empty). Callers in
    build_assets should treat None as 'feature unavailable for this run' and
    omit the `attention` key from the activations payload — the frontend
    handles that gracefully (per ADR-0008).
    """
    if "attn" not in captured:
        return None
    attn = captured["attn"]                          # [B, heads, N, N], cpu tensor
    attn_np = attn[0].numpy().astype(np.float32)     # drop batch -> [heads, N, N]
    quantized = quantize_heads(attn_np)
    return {
        "idx": idx,
        "path": path,
        "heads": int(attn_np.shape[0]),
        "gridH": int(captured["H"]),
        "gridW": int(captured["W"]),
        **quantized,  # dtype, shape, encoding, quantization, min, max, data
    }
