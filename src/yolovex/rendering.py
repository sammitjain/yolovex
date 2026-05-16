"""Shared rendering helpers — viridis heatmap encoding for activation thumbnails."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401  — keep Agg backend pin honored
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TOP_K_CHANNELS = 16
TOP_K_CLASSES = 6
CHANNEL_MAX_DIM = 80
CLASS_MAX_DIM = 60
JPEG_QUALITY = 85

_VIRIDIS = matplotlib.colormaps["viridis"]


def _capped_size(h: int, w: int, max_dim: int) -> tuple[int, int]:
    """(W, H) for PIL.resize, preserving aspect, capped at max_dim, never upscaling."""
    scale = min(max_dim / max(h, w), 1.0)
    return (max(1, int(round(w * scale))), max(1, int(round(h * scale))))


def _heatmap_to_b64(arr_2d: np.ndarray, max_dim: int) -> str:
    """Render a 2D array as a viridis heatmap JPEG, return as base64 data URI.

    Renders at min(native, max_dim) — CSS upscales for display. JPEG @ q=85
    is ~30x smaller than PNG for smooth viridis gradients with no visible loss.
    """
    a = arr_2d.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    a_norm = (a - lo) / (hi - lo + 1e-8)
    rgb = (_VIRIDIS(a_norm)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    h, w = arr_2d.shape
    target = _capped_size(h, w, max_dim)
    if target != (w, h):
        img = img.resize(target, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
