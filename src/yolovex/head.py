"""Detection-head visualization.

yolo26's Detect head returns ``(preds, info_dict)`` where:
  preds       : (B, max_det, 6) post-NMS boxes — (x1, y1, x2, y2, conf, cls)
  info_dict   : {'one2many': ..., 'one2one': {'boxes', 'scores', 'feats'}}

`scores` is (B, nc, total_anchors). Anchors decompose row-major across the
per-scale feature maps in `feats` (P3, P4, P5 at strides 8/16/32). We split
and reshape into per-scale `(nc, H, W)` class-score grids — those are the
"where on the image does the model think class C is" heatmaps that anchor
the whole explanation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image  # noqa: E402

CMAP_HEAT = "magma"
SCALE_NAMES = ["P3 (stride 8 — small objects)", "P4 (stride 16 — medium)", "P5 (stride 32 — large)"]


def split_scores_by_scale(scores: torch.Tensor, feats: list[torch.Tensor]) -> list[torch.Tensor]:
    """scores: (1, nc, total_anchors). Returns list of (nc, H, W) per scale."""
    out = []
    offset = 0
    for f in feats:
        _, _, H, W = f.shape
        n = H * W
        chunk = scores[0, :, offset : offset + n].reshape(scores.shape[1], H, W)
        out.append(chunk)
        offset += n
    return out


def _overlay(image_pil: Image.Image, heatmap_2d: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Resize heatmap to image dims and alpha-blend with the image. Returns RGB uint8."""
    W, H = image_pil.size
    h = heatmap_2d.astype(np.float32)
    lo, hi = float(h.min()), float(h.max())
    h_norm = (h - lo) / (hi - lo + 1e-8)
    h_img = Image.fromarray((h_norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    h_arr = np.asarray(h_img, dtype=np.float32) / 255.0
    colored = cm.get_cmap(CMAP_HEAT)(h_arr)[..., :3]
    base = np.asarray(image_pil.convert("RGB"), dtype=np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * colored
    return (blended * 255).astype(np.uint8)


def _draw_boxes(ax, boxes_xyxy, labels, colors=None) -> None:
    if colors is None:
        colors = ["#00ff88"] * len(boxes_xyxy)
    for (x1, y1, x2, y2), label, c in zip(boxes_xyxy, labels, colors):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=2))
        ax.text(
            x1, max(0, y1 - 4), label, color="black",
            fontsize=8, bbox=dict(facecolor=c, alpha=0.8, pad=1, edgecolor="none"),
        )


def _detected_top_classes(boxes: torch.Tensor, conf_thresh: float, k: int) -> list[tuple[int, float]]:
    """Return list of (class_id, peak_conf) for distinct detected classes, sorted by conf."""
    if boxes.numel() == 0:
        return []
    keep = boxes[:, 4] > conf_thresh
    if not keep.any():
        return []
    kept = boxes[keep]
    seen: dict[int, float] = {}
    for x1, y1, x2, y2, conf, cls in kept.tolist():
        cid = int(cls)
        if conf > seen.get(cid, -1.0):
            seen[cid] = float(conf)
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:k]


def render_predictions(
    results,
    head_output,
    image_path: Path,
    class_names: dict,
    out_path: Path,
    conf_thresh: float = 0.25,
    top_classes: int = 4,
) -> Path:
    """One big panel: input+boxes, per-scale max-class heatmap (P3/P4/P5),
    plus per-class heatmaps for top detected classes."""
    preds, info = head_output
    branch = info["one2one"]
    scores = branch["scores"].detach().float().cpu()      # (1, nc, total)
    feats = [f.detach() for f in branch["feats"]]
    per_scale_class = split_scores_by_scale(scores, feats)  # list of (nc, H, W)

    img = Image.open(image_path).convert("RGB")
    res = results[0]
    # Boxes scaled to original image coords by Ultralytics:
    box_arr = res.boxes.data.detach().cpu() if res.boxes is not None else torch.empty(0, 6)

    top = _detected_top_classes(box_arr, conf_thresh, top_classes)
    n_panels = 1 + 3 + len(top)
    cols = 4
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes = np.array(axes).reshape(-1)

    # Panel 0: input + boxes
    axes[0].imshow(img)
    if len(box_arr) > 0:
        keep = box_arr[:, 4] > conf_thresh
        if keep.any():
            kept = box_arr[keep]
            labels = [f"{class_names[int(c)]} {conf:.2f}" for *_, conf, c in kept.tolist()]
            _draw_boxes(axes[0], [b[:4].tolist() for b in kept], labels)
    axes[0].set_title(f"input + final detections (conf > {conf_thresh})", fontsize=10)
    axes[0].axis("off")

    # Panels 1..3: per-scale max-class score heatmap
    for i, (per_scale, name) in enumerate(zip(per_scale_class, SCALE_NAMES)):
        max_map = per_scale.max(dim=0).values.numpy()      # (H, W)
        H, W = max_map.shape
        overlay = _overlay(img, max_map, alpha=0.55)
        axes[1 + i].imshow(overlay)
        axes[1 + i].set_title(
            f"{name}\nmax class score over {per_scale.shape[0]} classes — grid {H}×{W}",
            fontsize=9,
        )
        axes[1 + i].axis("off")

    # Panels 4+: per-class heatmaps for top detected classes (max across scales after upsample)
    for i, (cid, peak) in enumerate(top):
        # Take the highest-resolution scale (P3) for this class — gives sharpest heatmap
        cls_map = per_scale_class[0][cid].numpy()
        overlay = _overlay(img, cls_map, alpha=0.6)
        ax = axes[4 + i]
        ax.imshow(overlay)
        ax.set_title(
            f"class '{class_names[cid]}' (id {cid})\nP3 heatmap — peak conf {peak:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Detection head — what the model decides, and where",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
