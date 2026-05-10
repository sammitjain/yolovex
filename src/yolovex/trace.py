"""Spatial-location trace.

For a given (x, y) on the input image, follow that location through the
network: at every block whose output is (B, C, H, W), find the corresponding
cell in the lower-resolution grid, summarize the C-vector there (norm, mean
|act|), and chart how the network's response at that location grows with
depth.

Output is a single 3-row figure:
  row 1: input image with a marker at (x, y)
  row 2: bar chart of channel-mean |activation| at (x, y) per layer
  row 3: filmstrip of each layer's mean-over-channels map with the
         scaled (x, y) marker overlaid — visual context for the bar above.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def _scale_xy(x: int, y: int, img_w: int, img_h: int, blk_w: int, blk_h: int) -> tuple[int, int]:
    bx = min(blk_w - 1, max(0, int(round(x * blk_w / img_w))))
    by = min(blk_h - 1, max(0, int(round(y * blk_h / img_h))))
    return bx, by


def render_trace(
    activations: dict, blocks: list, image_path: Path, x: int, y: int, out_path: Path
) -> Path:
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    if not (0 <= x < img_w and 0 <= y < img_h):
        raise ValueError(f"(x, y) = ({x}, {y}) outside image dims {img_w}×{img_h}")

    # Collect tensor-valued blocks in execution order
    rows = []
    for i, blk in enumerate(blocks):
        a = activations.get(i)
        if torch.is_tensor(a) and a.dim() == 4:
            t = a.detach().float().cpu()[0]  # (C, H, W)
            C, H, W = t.shape
            bx, by = _scale_xy(x, y, img_w, img_h, W, H)
            vec = t[:, by, bx]                # (C,)
            rows.append({
                "idx": i, "type": type(blk).__name__,
                "tensor": t, "C": C, "H": H, "W": W,
                "bx": bx, "by": by,
                "mean_abs": float(vec.abs().mean()),
                "l2": float(vec.norm()),
            })

    n = len(rows)
    fig = plt.figure(figsize=(max(12, n * 0.9), 9))
    gs = fig.add_gridspec(3, n, height_ratios=[2.2, 1.4, 1.6], hspace=0.35)

    # Row 1: input + marker (spans all columns)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img)
    ax_img.scatter([x], [y], s=160, c="red", edgecolor="white", linewidth=1.8, zorder=5)
    ax_img.set_title(f"input — tracing location (x={x}, y={y})", fontsize=11)
    ax_img.axis("off")

    # Row 2: bar chart of mean |activation| at (x, y) per layer
    ax_bar = fig.add_subplot(gs[1, :])
    xs = np.arange(n)
    vals = [r["mean_abs"] for r in rows]
    ax_bar.bar(xs, vals, color="#3b82f6")
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([f"{r['idx']}\n{r['type']}" for r in rows], fontsize=7, rotation=0)
    ax_bar.set_ylabel("mean |activation|\nat (x,y) across channels", fontsize=9)
    ax_bar.set_title("how strongly the network represents this location, per layer", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.25)

    # Row 3: filmstrip of mean-channel maps with marker
    for col, r in enumerate(rows):
        ax = fig.add_subplot(gs[2, col])
        m = r["tensor"].mean(dim=0).numpy()
        ax.imshow(m, cmap="viridis")
        ax.scatter([r["bx"]], [r["by"]], s=22, c="red", edgecolor="white", linewidth=0.6, zorder=5)
        ax.set_title(f"[{r['idx']}] {r['H']}×{r['W']}", fontsize=7)
        ax.axis("off")

    fig.suptitle("Spatial-location trace — same point, all depths", fontsize=12, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
