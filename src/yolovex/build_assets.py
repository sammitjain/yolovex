"""Generate frontend/data.js from real model inference.

Runs YOLO26n on the configured image, captures activations via hooks,
renders all per-block channel heatmaps + per-class detection heatmaps as
base64-encoded PNGs, and emits a single data.js file the React app reads
as `window.YV_DATA`.

Output structure:
{
  meta: { weights, image, imgsz, image_w, image_h },
  blocks: {
    0: {
      type, role, shape,
      sources: [-1] | [<int>, ...],
      input_shapes: [[1, 3, H, W], ...],
      desc, params,
      channel_pngs: ["data:image/png;base64,...", ...],   # top-K by |activation|
      channel_indices: [<original-channel-index>, ...],
      total_channels: <int>,
      mean_thumbnail: "data:image/png;base64,...",
      stats: { min, max, mean, std },
    },
    ...
    23: {
      type: "Detect", role: "Head",
      sources: [16, 19, 22],
      boxes: [{x1,y1,x2,y2,conf,cls_id,cls_name}, ...],   # normalized to [0,1]
      scales: { P3: {...}, P4: {...}, P5: {...} },
    }
  }
}
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from .activations import capture_with_results
from .head import split_scores_by_scale
from .layers import BLOCK_DESCRIPTIONS, build_layer_table
from .model import DEFAULT_WEIGHTS, load_model
from .topology import extract_topology

CMAP = "viridis"
TOP_K_CHANNELS = 16
TOP_K_CLASSES = 6
# Render at native tensor resolution capped at this max dim (CSS upscales).
# Smooth gradients compress excellently as JPEG; PNG was 50x bigger.
CHANNEL_MAX_DIM = 80
CLASS_MAX_DIM = 60
JPEG_QUALITY = 85
BOX_CONF_MIN = 0.05                   # only emit boxes above this (UI filters further)


_VIRIDIS = matplotlib.colormaps["viridis"]


def _capped_size(h: int, w: int, max_dim: int) -> tuple[int, int]:
    """(W, H) for PIL.resize, preserving aspect, capped at max_dim, never upscaling."""
    scale = min(max_dim / max(h, w), 1.0)
    return (max(1, int(round(w * scale))), max(1, int(round(h * scale))))


def _heatmap_to_b64(arr_2d: np.ndarray, max_dim: int) -> str:
    """Render a 2D array as a viridis heatmap JPEG, return as base64 data URI.

    Renders at min(native, max_dim) — CSS upscales for display. JPEG @ q=85
    is ~30× smaller than PNG for smooth viridis gradients with no visible loss.
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


def _extract_internals(block) -> dict:
    """Pull type-specific structural details out of a block module — used by the
    frontend to draw accurate inside-the-block diagrams (kernel size, sub-block
    type, count of bottlenecks, etc.)."""
    t = type(block).__name__
    info: dict = {"type": t}
    try:
        if t == "Conv":
            info.update({
                "kernel_size": list(block.conv.kernel_size),
                "stride": list(block.conv.stride),
                "in_channels": int(block.conv.in_channels),
                "out_channels": int(block.conv.out_channels),
            })
        elif t in {"C3k2", "C3", "C2f"}:
            sub_type = type(block.m[0]).__name__ if len(block.m) else "?"
            info.update({
                "n": len(block.m),
                "in_channels": int(block.cv1.conv.in_channels),
                "out_channels": int(block.cv2.conv.out_channels),
                "sub_block": sub_type,
            })
        elif t == "SPPF":
            info.update({
                "pool_kernel": int(block.m.kernel_size),
                "n_pools": 3,
                "in_channels": int(block.cv1.conv.in_channels),
                "out_channels": int(block.cv2.conv.out_channels),
            })
        elif t == "C2PSA":
            sub_type = type(block.m[0]).__name__ if len(block.m) else "?"
            info.update({
                "n": len(block.m),
                "in_channels": int(block.cv1.conv.in_channels),
                "out_channels": int(block.cv2.conv.out_channels),
                "sub_block": sub_type,
            })
        elif t == "Upsample":
            info.update({
                "scale_factor": float(block.scale_factor),
                "mode": str(block.mode),
            })
        elif t == "Concat":
            info.update({"axis": int(block.d)})
        elif t == "Detect":
            info.update({
                "nc": int(block.nc),
                "reg_max": int(block.reg_max),
                "no": int(block.no),
                "strides": [float(s) for s in block.stride.tolist()],
            })
    except Exception:
        pass
    return info


def _render_block(idx: int, tensor: torch.Tensor) -> dict:
    """Render channel PNGs + stats for one block's (B, C, H, W) tensor."""
    t = tensor.detach().float().cpu()[0]  # (C, H, W)
    C, _, _ = t.shape

    scores = t.abs().mean(dim=(1, 2))
    order = torch.argsort(scores, descending=True)[: TOP_K_CHANNELS].tolist()

    return {
        "channel_pngs": [_heatmap_to_b64(t[ch].numpy(), CHANNEL_MAX_DIM) for ch in order],
        "channel_indices": order,
        "total_channels": int(C),
        "mean_thumbnail": _heatmap_to_b64(t.mean(dim=0).numpy(), CHANNEL_MAX_DIM),
        "stats": {
            "min": round(float(t.min()), 3),
            "max": round(float(t.max()), 3),
            "mean": round(float(t.mean()), 4),
            "std": round(float(t.std()), 4),
        },
    }


def _render_detect(head_output, image_pil: Image.Image, names: dict, results) -> dict:
    """Render per-scale per-class heatmaps + box coords (normalized)."""
    _preds, info = head_output
    branch = info["one2one"]
    # head outputs raw class logits — apply sigmoid to get [0,1] probabilities,
    # so heatmaps colormap-normalize sensibly and `peak` matches NMS confidence.
    scores = torch.sigmoid(branch["scores"].detach().float().cpu())  # (1, nc, total_anchors)
    feats = [f.detach() for f in branch["feats"]]
    per_scale = split_scores_by_scale(scores, feats)  # list of (nc, H, W) per scale

    # Pick top classes by max peak across all scales (so the same set appears in every row)
    all_peaks = torch.stack([s.amax(dim=(1, 2)) for s in per_scale], dim=0)  # (3, nc)
    overall_peak = all_peaks.amax(dim=0)                                      # (nc,)
    top_class_ids = torch.argsort(overall_peak, descending=True)[: TOP_K_CLASSES].tolist()

    scale_specs = [
        ("P3", 8, "small objects"),
        ("P4", 16, "medium objects"),
        ("P5", 32, "large objects"),
    ]
    scales: dict[str, dict] = {}
    for (name, stride, size_label), ps in zip(scale_specs, per_scale):
        nc, H, W = ps.shape
        classes = []
        for cid in top_class_ids:
            class_map = ps[cid].numpy()
            classes.append({
                "id": int(cid),
                "name": names[int(cid)],
                "peak": round(float(class_map.max()), 3),
                "png": _heatmap_to_b64(class_map, CLASS_MAX_DIM),
            })
        scales[name] = {
            "stride": stride,
            "size_label": size_label,
            "grid_w": W,
            "grid_h": H,
            "classes": classes,
        }

    # Boxes from results — already scaled to original image coords by Ultralytics
    img_w, img_h = image_pil.size
    boxes = []
    res = results[0]
    if res.boxes is not None and len(res.boxes) > 0:
        for x1, y1, x2, y2, conf, cls in res.boxes.data.detach().cpu().tolist():
            if conf < BOX_CONF_MIN:
                continue
            boxes.append({
                "x1": round(x1 / img_w, 4),
                "y1": round(y1 / img_h, 4),
                "x2": round(x2 / img_w, 4),
                "y2": round(y2 / img_h, 4),
                "conf": round(float(conf), 3),
                "cls_id": int(cls),
                "cls_name": names[int(cls)],
            })
    boxes.sort(key=lambda b: b["conf"], reverse=True)
    return {"boxes": boxes, "scales": scales}


def build(image_path: Path, weights: str = DEFAULT_WEIGHTS, imgsz: int = 640) -> dict:
    yolo = load_model(weights)
    results, activations, blocks = capture_with_results(yolo, image_path, imgsz=imgsz)
    nodes, _edges = extract_topology(blocks, activations)
    rows = build_layer_table(blocks, activations)
    role_for = {r.index: r.role for r in rows}

    image_pil = Image.open(image_path).convert("RGB")
    img_w, img_h = image_pil.size

    # The model sees a *letterboxed* input, not the raw image. Derive it from
    # block 0's output (stride-2 conv → 2× the spatial dims). This is what we
    # report as the "model input" shape so the math through the network stays
    # consistent (640/2 = 320, etc.).
    block0_act = activations.get(0)
    if torch.is_tensor(block0_act) and block0_act.dim() == 4:
        _, _, h0, w0 = block0_act.shape
        model_input_shape = [1, 3, h0 * 2, w0 * 2]
    else:
        model_input_shape = [1, 3, imgsz, imgsz]

    blocks_data: dict[int, dict] = {}
    for node in nodes:
        a = activations.get(node.idx)
        block_data: dict = {
            "type": node.type_name,
            "role": "Head" if role_for.get(node.idx) == "head" else (
                "Backbone" if node.idx <= 10 else "Neck"
            ),
            "sources": list(node.sources),
            "desc": BLOCK_DESCRIPTIONS.get(node.type_name, "—"),
            "internals": _extract_internals(blocks[node.idx]),
        }

        # input_shapes via topology — for each source, look up its activation shape.
        # Source -1 == "the model's input tensor", which is the letterboxed image,
        # not the raw image dims.
        input_shapes = []
        for src in node.sources:
            if src < 0:
                input_shapes.append(model_input_shape)
                continue
            src_a = activations.get(src)
            if torch.is_tensor(src_a):
                input_shapes.append(list(src_a.shape))
            else:
                input_shapes.append(None)
        block_data["input_shapes"] = input_shapes

        if torch.is_tensor(a) and a.dim() == 4:
            block_data["shape"] = list(a.shape)
            block_data.update(_render_block(node.idx, a))
        elif node.type_name == "Detect":
            block_data["shape"] = "multi"
            block_data.update(_render_detect(a, image_pil, yolo.names, results))

        blocks_data[node.idx] = block_data

    return {
        "meta": {
            "weights": weights,
            "image": str(image_path),
            "imgsz": imgsz,
            "image_w": img_w,
            "image_h": img_h,
        },
        "blocks": blocks_data,
    }


def write_data_js(data: dict, out_path: Path) -> Path:
    """Write `window.YV_DATA = {...};` JS file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("window.YV_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    return out_path
