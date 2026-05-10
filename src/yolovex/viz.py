"""Activation rendering. Three modes — channel mean, single channel, top-K grid.

All renderers write a PNG to disk and return the path. Titles are intentionally
verbose so the saved images stand on their own without context.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI backend; we only save files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CMAP = "viridis"


def _prepare(tensor: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) or (C, H, W) → (C, H, W) float CPU tensor, autograd-detached."""
    t = tensor.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"expected (1,C,H,W) or (C,H,W); got shape {tuple(tensor.shape)}")
    return t


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def render_mean(tensor: torch.Tensor, layer_idx: int, type_name: str, out_path: Path) -> Path:
    t = _prepare(tensor)
    C, H, W = t.shape
    summary = t.mean(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(summary, cmap=CMAP)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Layer {layer_idx} ({type_name}) — channel mean over {C} channels — spatial {H}×{W}",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, out_path)


def render_channel(
    tensor: torch.Tensor, layer_idx: int, type_name: str, channel: int, out_path: Path
) -> Path:
    t = _prepare(tensor)
    C, H, W = t.shape
    if not (0 <= channel < C):
        raise IndexError(
            f"channel {channel} out of range; layer {layer_idx} has {C} channels (0..{C - 1})"
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(t[channel].numpy(), cmap=CMAP)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Layer {layer_idx} ({type_name}) — channel {channel} of {C} — spatial {H}×{W}",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, out_path)


def render_top_k(
    tensor: torch.Tensor, layer_idx: int, type_name: str, k: int, out_path: Path
) -> Path:
    t = _prepare(tensor)
    C, H, W = t.shape
    k = max(1, min(k, C))

    scores = t.abs().mean(dim=(1, 2))
    order = torch.argsort(scores, descending=True)[:k].tolist()

    cols = min(k, 8)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8 + 0.6))
    axes = np.array(axes).reshape(-1)

    for ax, ch in zip(axes, order):
        ax.imshow(t[ch].numpy(), cmap=CMAP)
        ax.set_title(f"channel {ch}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(order):]:
        ax.axis("off")

    fig.suptitle(
        f"Layer {layer_idx} ({type_name}) — top {k} of {C} channels by mean |activation| — spatial {H}×{W}",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, out_path)
