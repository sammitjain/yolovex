"""Generate frontend/activations-v2.js — spec-driven, fully recursive activations.

Captures the output tensor of every fx node inside every L1 block by running a
torch.fx Interpreter on each block. The frontend (`frontend/v2/`) looks up
activations by L1 index + fx node name (which it already carries on every
subNode via `members[-1]`), so this matches the v2 spec exactly with no
per-type hardcoding.

Asset schema (window.YV_ACT):
    meta: { weights, image, imgsz, image_w, image_h, skipped: [idx, ...] }
    nodes: {
        "<idx>": {
            output: { shape, mean, topK, topIdx, totalChannels, stats },
            sub: {
                "<fx_node_name>": { shape, mean, topK, topIdx, totalChannels, stats },
                ...
            }
        }
    }

L1 block output is `nodes[idx].output`; any sub-node (whether revealed as a
leaf or aggregated) is `nodes[idx].sub[fx_name]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.fx

from .activations import capture_with_results
from .build_assets import (
    CHANNEL_MAX_DIM,
    CLASS_MAX_DIM,
    TOP_K_CHANNELS,
    _heatmap_to_b64,
)
from .build_assets_l2 import _preprocess_for_raw_forward
from .block_spec import _trace_with_patches
from .head import split_scores_by_scale
from .model import DEFAULT_WEIGHTS, get_blocks, load_model

# How many top classes (by overall peak score) to emit per scale in the Detect
# payload. Frontend lets the user choose how many of these to actually render
# (default 6, capped at this build-time value).
TOP_K_CLASSES_V2 = 12


def _render_tensor(tensor: torch.Tensor) -> dict | None:
    """Render top-K channel + mean JPEGs for a (B,C,H,W) tensor. None if shape not 4D."""
    if not torch.is_tensor(tensor) or tensor.dim() != 4:
        return None
    t = tensor.detach().float().cpu()[0]
    C = t.shape[0]
    scores = t.abs().mean(dim=(1, 2))
    order = torch.argsort(scores, descending=True)[:TOP_K_CHANNELS].tolist()
    return {
        "shape": list(tensor.shape),
        "mean": _heatmap_to_b64(t.mean(dim=0).numpy(), CHANNEL_MAX_DIM),
        "topK": [_heatmap_to_b64(t[ch].numpy(), CHANNEL_MAX_DIM) for ch in order],
        "topIdx": order,
        "totalChannels": int(C),
        "stats": {
            "min": round(float(t.min()), 3),
            "max": round(float(t.max()), 3),
            "mean": round(float(t.mean()), 4),
            "std": round(float(t.std()), 4),
        },
    }


class _CapturingInterpreter(torch.fx.Interpreter):
    """Runs the graph and stashes every node's output value by fx name."""

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.captured: dict[str, Any] = {}

    def run_node(self, n):
        out = super().run_node(n)
        self.captured[n.name] = out
        return out


def _capture_block_inputs(blocks: list) -> tuple[dict[int, Any], list]:
    """Register forward pre-hooks on every L1 block, capturing its input tensor."""
    inputs: dict[int, Any] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, args):
            # args is the positional tuple; for these blocks it's (x,) where x is
            # a Tensor or a list of Tensors (Concat). We keep whatever it is.
            inputs[idx] = args[0] if len(args) == 1 else args
        return hook

    for idx, b in enumerate(blocks):
        handles.append(b.register_forward_pre_hook(make_hook(idx)))
    return inputs, handles


def _capture_block_outputs(blocks: list) -> tuple[dict[int, Any], list]:
    """Forward hooks for L1 outputs (used when fx trace fails, e.g. Concat)."""
    outputs: dict[int, Any] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            outputs[idx] = out
        return hook

    for idx, b in enumerate(blocks):
        handles.append(b.register_forward_hook(make_hook(idx)))
    return outputs, handles


def _build_detect_payload(
    image_path: Path,
    weights: str,
    imgsz: int,
    top_k_classes: int = TOP_K_CLASSES_V2,
) -> dict | None:
    """Run an Ultralytics predict pass to harvest the Detect head's per-scale
    class score maps + survivors + low-conf candidate boxes.

    Returns a dict shaped:
        {
          boxes:           [...post-NMS survivors (conf>=0.05)],
          candidate_boxes: [...up to ~50 low-conf candidates (conf>=0.001)],
          scales:          { P3: {stride, grid_w, grid_h, classes:[{id,name,peak,png}]}, P4, P5 },
          nc, names, strides,
        }
    Returns None on failure (non-fatal — frontend falls back to empty Detect panel).
    """
    try:
        from PIL import Image as _PILImage
    except Exception:
        return None

    payload: dict[str, Any] = {}

    # Pass 1: high-level predict path (captures both head output for heatmaps and
    # post-NMS survivors). This calls model.fuse() internally — fine, we only
    # need the head's raw scores + boxes from this pass.
    try:
        yolo = load_model(weights)
        results, activations, _blocks = capture_with_results(yolo, image_path, imgsz=imgsz)
        # Find the Detect block's activation entry (head output tuple).
        head_act = None
        for k, v in activations.items():
            # Detect activation is a tuple (_preds, info), not a 4-D tensor.
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict) and "one2one" in v[1]:
                head_act = v
                break
        if head_act is None:
            print("  [v2] could not locate Detect head activation; skipping detect payload")
            return None

        _preds, info = head_act
        branch = info["one2one"]
        scores = torch.sigmoid(branch["scores"].detach().float().cpu())  # (1, nc, A)
        feats = [f.detach() for f in branch["feats"]]
        per_scale = split_scores_by_scale(scores, feats)  # list of (nc, H, W)

        all_peaks = torch.stack([s.amax(dim=(1, 2)) for s in per_scale], dim=0)
        overall_peak = all_peaks.amax(dim=0)  # (nc,)
        top_ids = torch.argsort(overall_peak, descending=True)[: top_k_classes].tolist()

        names = yolo.names
        scale_specs = [
            ("P3", 8, "small objects"),
            ("P4", 16, "medium objects"),
            ("P5", 32, "large objects"),
        ]
        scales: dict[str, dict] = {}
        for (sname, stride, size_label), ps in zip(scale_specs, per_scale):
            _nc, H, W = ps.shape
            classes = []
            for cid in top_ids:
                cmap = ps[cid].numpy()
                classes.append({
                    "id": int(cid),
                    "name": names[int(cid)],
                    "peak": round(float(cmap.max()), 3),
                    "png": _heatmap_to_b64(cmap, CLASS_MAX_DIM),
                })
            scales[sname] = {
                "stride": stride,
                "size_label": size_label,
                "grid_w": int(W),
                "grid_h": int(H),
                "classes": classes,
            }

        with _PILImage.open(image_path) as im:
            img_w, img_h = im.size

        boxes = []
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            for x1, y1, x2, y2, conf, cls_ in res.boxes.data.detach().cpu().tolist():
                if conf < 0.05:
                    continue
                boxes.append({
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4),
                    "conf": round(float(conf), 3),
                    "cls_id": int(cls_),
                    "cls_name": names[int(cls_)],
                })
        boxes.sort(key=lambda b: b["conf"], reverse=True)

        payload["boxes"] = boxes
        payload["scales"] = scales
        payload["nc"] = int(scores.shape[1])
        payload["names"] = {int(k): v for k, v in names.items()}
        payload["strides"] = [8, 16, 32]
    except Exception as e:
        print(f"  [v2] detect main capture failed: {e.__class__.__name__}: {e}")
        return None

    # Pass 2: low-conf candidate boxes — separate predict at conf=0.001 to surface
    # runners-up the user can reveal with the slider.
    try:
        yolo_low = load_model(weights)
        results_low = yolo_low(str(image_path), imgsz=imgsz, conf=0.001, verbose=False)
        with _PILImage.open(image_path) as im:
            img_w, img_h = im.size
        candidate_boxes = []
        res = results_low[0]
        if res.boxes is not None and len(res.boxes) > 0:
            names_low = yolo_low.names
            for x1, y1, x2, y2, conf, cls_ in res.boxes.data.detach().cpu().tolist():
                candidate_boxes.append({
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4),
                    "conf": round(float(conf), 4),
                    "cls_id": int(cls_),
                    "cls_name": names_low[int(cls_)],
                })
            candidate_boxes.sort(key=lambda b: b["conf"], reverse=True)
        payload["candidate_boxes"] = candidate_boxes
        print(f"  [v2] detect: {len(payload['boxes'])} survivors, {len(candidate_boxes)} candidates")
    except Exception as e:
        print(f"  [v2] candidate-box capture failed (non-fatal): {e}")
        payload.setdefault("candidate_boxes", [])

    return payload


def build(image_path: Path, weights: str = DEFAULT_WEIGHTS, imgsz: int = 640) -> dict:
    print(f"  [v2] loading {weights}...")
    yolo = load_model(weights)
    blocks = get_blocks(yolo)

    input_tensor = _preprocess_for_raw_forward(image_path, imgsz)

    print(f"  [v2] running unfused forward to capture per-block inputs/outputs...")
    block_inputs, in_handles = _capture_block_inputs(blocks)
    block_outputs, out_handles = _capture_block_outputs(blocks)
    try:
        with torch.no_grad():
            yolo.model(input_tensor)
    finally:
        for h in in_handles + out_handles:
            h.remove()

    skipped: list[int] = []
    detect_indices: list[int] = []
    nodes_out: dict[str, dict] = {}

    print(f"  [v2] re-interpreting each block's fx graph to harvest sub-activations...")
    for idx, block in enumerate(blocks):
        cls = type(block).__name__
        key = str(idx)
        entry: dict[str, Any] = {"type": cls, "output": None, "sub": {}}

        if cls == "Detect":
            # Detect's fx graph isn't a clean 4-D in/out — handle it separately
            # below (per-scale heatmaps + boxes + candidate_boxes).
            detect_indices.append(idx)
            nodes_out[key] = entry
            print(f"    [{idx:>2}] {cls:<10} (detect payload built after main loop)")
            continue

        # L1 output (always available from forward hook)
        l1_out = block_outputs.get(idx)
        l1_render = _render_tensor(l1_out) if torch.is_tensor(l1_out) else None
        entry["output"] = l1_render

        # Sub-node capture via fx interpreter — best effort
        in_t = block_inputs.get(idx)
        sub_count = 0
        if torch.is_tensor(in_t):
            try:
                gm = _trace_with_patches(block)
                interp = _CapturingInterpreter(gm)
                with torch.no_grad():
                    interp.run(in_t)
                for n in gm.graph.nodes:
                    if n.op in ("placeholder", "output", "get_attr"):
                        continue
                    val = interp.captured.get(n.name)
                    rendered = _render_tensor(val) if torch.is_tensor(val) else None
                    if rendered is not None:
                        entry["sub"][n.name] = rendered
                        sub_count += 1
            except Exception as e:
                print(f"    [{idx:>2}] {cls:<10} fx trace/interp failed ({e.__class__.__name__}: {e}) — L1 only")

        nodes_out[key] = entry
        print(f"    [{idx:>2}] {cls:<10} L1{'+' if l1_render else '-'} sub={sub_count}")

    # Detect payload — single capture pass shared by all Detect blocks in the model.
    if detect_indices:
        det = _build_detect_payload(image_path, weights, imgsz)
        if det is not None:
            for didx in detect_indices:
                nodes_out[str(didx)]["detect"] = det
        else:
            # Capture failed — fall back to deferred behavior for Detect.
            for didx in detect_indices:
                skipped.append(didx)

    image_w = image_h = None
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(image_path) as im:
            image_w, image_h = im.size
    except Exception:
        pass

    return {
        "meta": {
            "weights": weights,
            "image": str(image_path),
            "imgsz": imgsz,
            "image_w": image_w,
            "image_h": image_h,
            "skipped": skipped,
        },
        "nodes": nodes_out,
    }


def write_activations_js(data: dict, out_path: Path) -> None:
    """Serialize as `window.YV_ACT = {...};`."""
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out_path.write_text(f"window.YV_ACT = {payload};\n", encoding="utf-8")
