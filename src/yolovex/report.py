"""Generate a self-contained HTML report.

One file, no JS, all images embedded as base64. Sections:
  1. Header — model, image, summary stats
  2. Final detections — input image with boxes
  3. Head heatmaps — per-scale max-class score overlays + top-class overlays
  4. Architecture flow — every block as a card with thumbnail + metadata
  5. Top-K detail per block (mean + top-8 channels)

Pure stdlib + matplotlib. Render once, share, archive.
"""

from __future__ import annotations

import base64
import io
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .head import _overlay, split_scores_by_scale  # noqa: F401
from .layers import LayerInfo, build_layer_table


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _block_thumbnail_b64(tensor: torch.Tensor, size_inches: float = 1.6) -> str:
    """Mean-over-channels thumbnail."""
    t = tensor.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    m = t.mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(size_inches, size_inches))
    ax.imshow(m, cmap="viridis")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return _fig_to_b64(fig)


def _block_topk_b64(tensor: torch.Tensor, k: int = 8) -> str:
    t = tensor.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    C = t.shape[0]
    k = min(k, C)
    scores = t.abs().mean(dim=(1, 2))
    order = torch.argsort(scores, descending=True)[:k].tolist()
    cols = min(k, 4)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax, ch in zip(axes, order):
        ax.imshow(t[ch].numpy(), cmap="viridis")
        ax.set_title(f"c{ch}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    return _fig_to_b64(fig)


def _input_b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _detections_b64(results, image_path: Path, conf: float = 0.25) -> str:
    """Render the input image with final boxes, return base64."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    res = results[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    if res.boxes is not None and len(res.boxes) > 0:
        box_arr = res.boxes.data.detach().cpu()
        keep = box_arr[:, 4] > conf
        names = results[0].names if hasattr(results[0], "names") else {}
        for x1, y1, x2, y2, c, cls in box_arr[keep].tolist():
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00ff88", linewidth=2)
            )
            label = f"{names.get(int(cls), cls)} {c:.2f}"
            ax.text(x1, max(0, y1 - 4), label, fontsize=8,
                    bbox=dict(facecolor="#00ff88", alpha=0.85, pad=1, edgecolor="none"))
    ax.set_title(f"final detections (conf > {conf})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _head_overlays_b64(head_output, image_path: Path) -> list[tuple[str, str]]:
    """Return list of (caption, base64-png) for per-scale max-class overlays."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    preds, info = head_output
    branch = info["one2one"]
    scores = branch["scores"].detach().float().cpu()
    feats = [f.detach() for f in branch["feats"]]
    per_scale = split_scores_by_scale(scores, feats)
    names = ["P3 (stride 8)", "P4 (stride 16)", "P5 (stride 32)"]
    out = []
    for ps, name in zip(per_scale, names):
        max_map = ps.max(dim=0).values.numpy()
        overlay = _overlay(img, max_map, alpha=0.55)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(overlay)
        H, W = max_map.shape
        ax.set_title(f"{name}\nmax class score — grid {H}×{W}", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        out.append((name, _fig_to_b64(fig)))
    return out


# ---------- HTML rendering ----------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px;
       margin: 24px auto; padding: 0 16px; color: #1f2328; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 16px; margin-top: 28px; padding-bottom: 6px; border-bottom: 1px solid #d0d7de; }
.meta { color: #59636e; font-size: 13px; margin-bottom: 18px; }
.row { display: flex; flex-wrap: wrap; gap: 12px; }
.card { border: 1px solid #d0d7de; border-radius: 6px; padding: 8px;
        background: #f6f8fa; width: 200px; font-size: 11px; }
.card.head { background: #fff8e6; }
.card img { width: 100%; display: block; border-radius: 3px; background: white; }
.card .idx { font-weight: bold; font-size: 13px; }
.card .type { color: #0969da; }
.card .role { float: right; color: #6f42c1; font-size: 10px; text-transform: uppercase; }
.card .desc { color: #59636e; margin-top: 4px; font-size: 10px; line-height: 1.3; }
.card .shape { font-family: ui-monospace, monospace; font-size: 10px; color: #1f2328; }
.detail { margin-top: 14px; }
.detail summary { cursor: pointer; font-weight: 600; color: #0969da; font-size: 13px; }
.detail-body { display: flex; gap: 16px; align-items: flex-start; padding-top: 10px;
               flex-wrap: wrap; }
.detail-body img { max-width: 100%; }
.input-img { max-width: 320px; border: 1px solid #d0d7de; border-radius: 4px; }
.head-row img { max-width: 320px; border: 1px solid #d0d7de; border-radius: 4px; }
"""


def _render_block_card(info: LayerInfo, thumb_b64: str | None) -> str:
    img_html = f'<img src="data:image/png;base64,{thumb_b64}"/>' if thumb_b64 else \
               '<div style="height:120px; display:flex; align-items:center; justify-content:center; color:#59636e;">non-tensor</div>'
    cls = "card head" if info.role == "head" else "card"
    return f"""
    <div class="{cls}">
      <span class="role">{info.role}</span>
      <div><span class="idx">[{info.index}]</span> <span class="type">{info.type_name}</span></div>
      <div class="shape">{info.output_shape}</div>
      <div class="desc">{info.description}</div>
      {img_html}
      <div style="font-size:10px;color:#59636e;margin-top:4px;">params: {info.n_params:,}</div>
    </div>
    """


def _render_block_detail(info: LayerInfo, topk_b64: str | None) -> str:
    if topk_b64 is None:
        return ""
    return f"""
    <details class="detail">
      <summary>[{info.index}] {info.type_name} — top-8 channels</summary>
      <div class="detail-body">
        <img src="data:image/png;base64,{topk_b64}"/>
      </div>
    </details>
    """


def generate_report(
    results, activations: dict, blocks: list, image_path: Path,
    weights: str, out_path: Path,
) -> Path:
    rows = build_layer_table(blocks, activations)

    # Per-block thumbnails + top-k details (only for tensor outputs)
    thumbs: dict[int, str] = {}
    topks: dict[int, str] = {}
    for r in rows:
        a = activations.get(r.index)
        if torch.is_tensor(a) and a.dim() == 4:
            thumbs[r.index] = _block_thumbnail_b64(a)
            topks[r.index] = _block_topk_b64(a)

    # Head & input
    input_b64 = _input_b64(image_path)
    detections_b64 = _detections_b64(results, image_path)
    head_overlays = _head_overlays_b64(activations[len(blocks) - 1], image_path) if isinstance(
        activations.get(len(blocks) - 1), tuple
    ) else []

    cards_html = "\n".join(_render_block_card(r, thumbs.get(r.index)) for r in rows)
    details_html = "\n".join(_render_block_detail(r, topks.get(r.index)) for r in rows)
    head_html = "\n".join(
        f'<div><img src="data:image/png;base64,{b64}"/></div>'
        for _, b64 in head_overlays
    )

    n_tensor = len(thumbs)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>yolovex report</title>
<style>{CSS}</style></head><body>
<h1>yolovex — architecture explainer</h1>
<div class="meta">
  weights: <code>{weights}</code> &nbsp;·&nbsp;
  image: <code>{image_path}</code> &nbsp;·&nbsp;
  blocks: {len(rows)} ({n_tensor} tensor outputs) &nbsp;·&nbsp;
  generated: {datetime.now().isoformat(timespec="seconds")}
</div>

<h2>1. Input &amp; final decision</h2>
<div class="row">
  <img class="input-img" src="data:image/jpeg;base64,{input_b64}"/>
  <img class="input-img" src="data:image/png;base64,{detections_b64}"/>
</div>

<h2>2. Detection head — where the model looks, per scale</h2>
<p style="font-size:12px;color:#59636e;">Bright = high max-class score across all 80 COCO classes at that grid cell.
P3 sees small objects (fine grid), P5 sees large ones (coarse grid).</p>
<div class="row head-row">
{head_html}
</div>

<h2>3. Architecture flow — every block, mean over channels</h2>
<p style="font-size:12px;color:#59636e;">Cards in execution order. Yellow = head. Each thumbnail is the channel-mean
activation for this image; expand a row below for the top-8 most active channels.</p>
<div class="row">
{cards_html}
</div>

<h2>4. Per-block detail (top channels)</h2>
{details_html}

</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
