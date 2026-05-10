"""Generate frontend/data-l2.js from real model inference.

Runs YOLO26n on the configured image, captures sub-module-granularity activations
via hooks, and emits a single data-l2.js file the React app reads as `window.YV_DATA`.

This is "Level 2": each top-level block is expanded into its direct sub-modules.
The resulting `blocks` map contains BOTH the parent block payloads (numeric string
keys "0", "4", "23") and sub-module entries ("0.conv", "4.cv1", "9.m.0", etc.).

Reuses helpers from build_assets.py — does NOT modify that file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .build_assets import (
    CHANNEL_MAX_DIM,
    JPEG_QUALITY,  # noqa: F401 — imported for completeness; _heatmap_to_b64 uses it
    TOP_K_CHANNELS,
    _capped_size,  # noqa: F401
    _extract_internals,
    _heatmap_to_b64,
    build as l1_build,
    write_data_js,
)
from .model import DEFAULT_WEIGHTS, get_blocks, load_model


def _preprocess_for_raw_forward(image_path, imgsz: int) -> torch.Tensor:
    """Letterbox an image and convert to a (1, 3, H, W) float tensor in [0, 1].

    Mirrors what Ultralytics' high-level predict path does for preprocessing,
    but lets us call the inner nn.Module directly (`yolo.model(tensor)`) so
    fusion never happens — leaving BatchNorm modules separate so their forward
    hooks fire normally.
    """
    from ultralytics.data.augment import LetterBox

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)
    img_lb = lb(image=img)
    return torch.from_numpy(np.ascontiguousarray(img_lb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# ---------------------------------------------------------------------------
# Expansion table: which sub-paths to hook per top-level block type
# ---------------------------------------------------------------------------

# Conv blocks (top-level): idx 0, 1, 3, 5, 7, 17, 20
CONV_EXPAND = [("conv", "Conv2d"), ("bn", "BatchNorm2d"), ("act", "SiLU")]

# C3k2 blocks: idx 2, 4, 6, 8, 13, 16, 19, 22
C3K2_HOOK_PATHS = ["cv1", "m.0", "cv2"]   # concat synthesized afterwards

# SPPF block: idx 9
SPPF_HOOK_PATHS = ["cv1", "cv2"]  # m hooked specially (3 calls)

# C2PSA block: idx 10
C2PSA_HOOK_PATHS = ["cv1", "m.0", "cv2"]

# Detect block: idx 23  -> synthesized per-scale references only (no hook)


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def _make_store_hook(store: dict, key: str):
    """Closure factory: stores the module's output tensor under `key`."""
    def hook(module, inp, out):
        store[key] = out
    return hook


def _make_sppf_m_hook(store: dict, counter: list, parent_idx: int):
    """Closure factory for SPPF's single MaxPool that's called 3 times.

    `counter` is a mutable list holding [call_index]. We increment it on
    each call so successive fires are stored as {parent_idx}.m.0, .m.1, .m.2.
    """
    def hook(module, inp, out):
        call_idx = counter[0]
        store[f"{parent_idx}.m.{call_idx}"] = out
        counter[0] += 1

    return hook


# ---------------------------------------------------------------------------
# Sub-module capture
# ---------------------------------------------------------------------------

def _resolve_submodule(block, path: str):
    """Walk dotted path (e.g. 'm.0') and return the sub-module."""
    obj = block
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def attach_l2_hooks(blocks: list) -> tuple[dict[str, Any], list]:
    """Register forward hooks on every relevant sub-module.

    Returns (sub_act dict, list of hook handles).
    sub_act will be populated after the forward pass.
    """
    sub_act: dict[str, Any] = {}
    handles = []

    for idx, block in enumerate(blocks):
        btype = type(block).__name__

        if btype == "Conv":
            # Hook conv (Conv2d) and bn (BatchNorm2d). Both fire during raw
            # `yolo.model(tensor)` inference because we explicitly avoid the
            # fusion that the high-level predict path triggers. act (SiLU) is
            # a single shared module across ALL Conv blocks, so we still rely
            # on the parent block's L1 activation for that one (act output ==
            # parent block output, since Conv.forward = act(bn(conv(x)))).
            for path in ("conv", "bn"):
                try:
                    sub = _resolve_submodule(block, path)
                    key = f"{idx}.{path}"
                    h = sub.register_forward_hook(_make_store_hook(sub_act, key))
                    handles.append(h)
                except (AttributeError, IndexError, TypeError):
                    pass

        elif btype == "C3k2":
            for path in C3K2_HOOK_PATHS:
                try:
                    sub = _resolve_submodule(block, path)
                    key = f"{idx}.{path}"
                    h = sub.register_forward_hook(_make_store_hook(sub_act, key))
                    handles.append(h)
                except (AttributeError, IndexError, TypeError):
                    pass

        elif btype == "SPPF":
            # cv1 and cv2 normally
            for path in SPPF_HOOK_PATHS:
                try:
                    sub = _resolve_submodule(block, path)
                    key = f"{idx}.{path}"
                    h = sub.register_forward_hook(_make_store_hook(sub_act, key))
                    handles.append(h)
                except (AttributeError, IndexError, TypeError):
                    pass
            # m fires 3 times — use special counter hook
            try:
                counter = [0]
                m_sub = _resolve_submodule(block, "m")
                h = m_sub.register_forward_hook(_make_sppf_m_hook(sub_act, counter, idx))
                handles.append(h)
            except (AttributeError, IndexError, TypeError):
                pass

        elif btype == "C2PSA":
            for path in C2PSA_HOOK_PATHS:
                try:
                    sub = _resolve_submodule(block, path)
                    key = f"{idx}.{path}"
                    h = sub.register_forward_hook(_make_store_hook(sub_act, key))
                    handles.append(h)
                except (AttributeError, IndexError, TypeError):
                    pass

        # Concat, Upsample, Bottleneck, PSABlock, Detect — no expansion at L2

    return sub_act, handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_tensor(tensor: torch.Tensor) -> dict:
    """Render channel PNGs + stats for a (B, C, H, W) tensor."""
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


def _synthesize_concat_c3k2(idx: int, sub_act: dict) -> None:
    """Synthesize 'idx.concat' for a C3k2 block.

    C3k2.forward: x1 = cv1(x)[half]; branches = [x1] + [m.0(x1)]; cat(branches) -> cv2
    The concat is torch.cat of cv1's second half and m.0's output.
    We approximate faithfully: cat([cv1_out[half_channels], m0_out]).
    """
    cv1_key = f"{idx}.cv1"
    m0_key = f"{idx}.m.0"
    cv1_t = sub_act.get(cv1_key)
    m0_t = sub_act.get(m0_key)
    if cv1_t is not None and m0_t is not None and torch.is_tensor(cv1_t) and torch.is_tensor(m0_t):
        # cv1 produces half channels passed into m; take the full output as is
        # then cat with m.0 output — matches C3k2's actual forward logic
        try:
            sub_act[f"{idx}.concat"] = torch.cat([cv1_t, m0_t], dim=1)
        except RuntimeError:
            pass


def _synthesize_concat_sppf(idx: int, sub_act: dict) -> None:
    """Synthesize 'idx.concat' for a SPPF block.

    SPPF.forward: y1=m(cv1); y2=m(y1); y3=m(y2); cat([cv1, y1, y2, y3]) -> cv2
    """
    cv1_key = f"{idx}.cv1"
    m0_key = f"{idx}.m.0"
    m1_key = f"{idx}.m.1"
    m2_key = f"{idx}.m.2"
    parts = [sub_act.get(k) for k in [cv1_key, m0_key, m1_key, m2_key]]
    if all(t is not None and torch.is_tensor(t) for t in parts):
        try:
            sub_act[f"{idx}.concat"] = torch.cat(parts, dim=1)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(image_path: Path, weights: str = DEFAULT_WEIGHTS, imgsz: int = 640) -> dict:
    """Build and return the L2 data dict (to be written by write_data_js)."""

    # Step 1: Run L1 builder to get parent block payloads + metadata
    print("  [L2] running L1 builder for parent block payloads...")
    l1_data = l1_build(image_path, weights=weights, imgsz=imgsz)

    # Step 1.5: Run a second predict pass at a much lower confidence threshold
    # to capture LOW-CONFIDENCE candidate boxes — the runners-up that the
    # one-to-one head emitted but that don't survive the user's display
    # threshold. Stored separately as `candidate_boxes` so the L2 frontend can
    # show them faded behind the final detections (with a UI slider).
    print("  [L2] capturing low-conf candidate boxes (conf=0.001)...")
    try:
        from PIL import Image as _PILImage
        yolo_low = load_model(weights)
        results_low = yolo_low(str(image_path), imgsz=imgsz, conf=0.001, verbose=False)
        img_pil = _PILImage.open(image_path).convert("RGB")
        img_w, img_h = img_pil.size
        candidate_boxes = []
        res = results_low[0]
        if res.boxes is not None and len(res.boxes) > 0:
            names = yolo_low.names
            for x1, y1, x2, y2, conf, cls in res.boxes.data.detach().cpu().tolist():
                candidate_boxes.append({
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4),
                    "conf": round(float(conf), 4),
                    "cls_id": int(cls),
                    "cls_name": names[int(cls)],
                })
            candidate_boxes.sort(key=lambda b: b["conf"], reverse=True)
        # Inject into the parent Detect block's payload (L1 path didn't have these)
        detect_payload = l1_data["blocks"].get(23) or l1_data["blocks"].get("23")
        if detect_payload is not None:
            detect_payload["candidate_boxes"] = candidate_boxes
        print(f"  [L2] captured {len(candidate_boxes)} candidate boxes (lowest conf {candidate_boxes[-1]['conf'] if candidate_boxes else '—'})")
    except Exception as e:
        print(f"  [L2] candidate-box capture failed (non-fatal): {e}")

    # Step 2: Load model FRESH (so it's unfused) and run RAW forward with L2 hooks.
    # Going through the high-level `yolo(image_path)` predictor would call
    # `model.fuse()` internally, folding BatchNorm into the preceding Conv2d
    # weights — at which point BN's forward hook never fires. We bypass the
    # predictor and call the inner nn.Module directly with a manually
    # preprocessed tensor.
    print("  [L2] attaching sub-module hooks and running raw forward (unfused)...")
    yolo = load_model(weights)
    blocks = get_blocks(yolo)

    input_tensor = _preprocess_for_raw_forward(image_path, imgsz)
    sub_act, handles = attach_l2_hooks(blocks)
    try:
        with torch.no_grad():
            yolo.model(input_tensor)
    finally:
        remove_hooks(handles)

    # Step 3: Synthesize implicit concat nodes
    for idx, block in enumerate(blocks):
        btype = type(block).__name__
        if btype == "C3k2":
            _synthesize_concat_c3k2(idx, sub_act)
        elif btype == "SPPF":
            _synthesize_concat_sppf(idx, sub_act)

    # Step 4: Build the blocks map — start with L1 parent payloads
    # Keys become string representations for consistent ID handling
    combined_blocks: dict[str, Any] = {}
    for k, v in l1_data["blocks"].items():
        combined_blocks[str(k)] = v

    # Step 5: Add sub-node entries
    for idx, block in enumerate(blocks):
        btype = type(block).__name__
        parent_str = str(idx)

        if btype == "Conv":
            # Expand into conv, bn, act.
            # - conv (Conv2d): fires normally during raw forward.
            # - bn (BatchNorm2d): fires now that we run unfused raw forward.
            # - act (SiLU): Ultralytics shares ONE SiLU instance across ALL Conv
            #   blocks, so hooking it would capture only the last Conv's call.
            #   We reuse the parent block's L1 activation (which IS act's
            #   output, since Conv.forward = act(bn(conv(x)))).
            parent_l1 = l1_data["blocks"].get(idx) or l1_data["blocks"].get(str(idx))

            for path, module_type in CONV_EXPAND:
                key = f"{idx}.{path}"
                entry: dict = {
                    "type": module_type,
                    "parent": idx,
                    "parent_type": "Conv",
                    "path": path,
                }
                if path == "act" and parent_l1 is not None:
                    entry["shape"] = parent_l1.get("shape")
                    entry["channel_pngs"] = parent_l1.get("channel_pngs", [])
                    entry["channel_indices"] = parent_l1.get("channel_indices", [])
                    entry["total_channels"] = parent_l1.get("total_channels", 0)
                    entry["mean_thumbnail"] = parent_l1.get("mean_thumbnail")
                    entry["stats"] = parent_l1.get("stats", {})
                    entry["reuses_parent_l1"] = True
                else:
                    t = sub_act.get(key)
                    if t is not None and torch.is_tensor(t) and t.dim() == 4:
                        entry["shape"] = list(t.shape)
                        entry.update(_render_tensor(t))
                    else:
                        entry["shape"] = None
                        entry["channel_pngs"] = []
                        entry["channel_indices"] = []
                        entry["total_channels"] = 0
                        entry["mean_thumbnail"] = None
                        entry["stats"] = {}
                combined_blocks[key] = entry

        elif btype == "C3k2":
            # cv1, m.0, concat (synthesized), cv2
            sub_entries = [
                ("cv1", "Conv"),
                ("m.0", type(block.m[0]).__name__),
                ("concat", "Concat"),
                ("cv2", "Conv"),
            ]
            for path, module_type in sub_entries:
                key = f"{idx}.{path}"
                t = sub_act.get(key)
                entry = {
                    "type": module_type,
                    "parent": idx,
                    "parent_type": "C3k2",
                    "path": path,
                }
                if path == "concat":
                    entry["synthesized"] = True
                if t is not None and torch.is_tensor(t) and t.dim() == 4:
                    entry["shape"] = list(t.shape)
                    entry.update(_render_tensor(t))
                else:
                    entry["shape"] = None
                    entry["channel_pngs"] = []
                    entry["channel_indices"] = []
                    entry["total_channels"] = 0
                    entry["mean_thumbnail"] = None
                    entry["stats"] = {}
                combined_blocks[key] = entry

        elif btype == "SPPF":
            # cv1, m.0, m.1, m.2, concat (synthesized), cv2
            sub_entries_sppf = [
                ("cv1", "Conv"),
                ("m.0", "MaxPool2d"),
                ("m.1", "MaxPool2d"),
                ("m.2", "MaxPool2d"),
                ("concat", "Concat"),
                ("cv2", "Conv"),
            ]
            for path, module_type in sub_entries_sppf:
                key = f"{idx}.{path}"
                t = sub_act.get(key)
                entry = {
                    "type": module_type,
                    "parent": idx,
                    "parent_type": "SPPF",
                    "path": path,
                }
                if path == "concat":
                    entry["synthesized"] = True
                if t is not None and torch.is_tensor(t) and t.dim() == 4:
                    entry["shape"] = list(t.shape)
                    entry.update(_render_tensor(t))
                else:
                    entry["shape"] = None
                    entry["channel_pngs"] = []
                    entry["channel_indices"] = []
                    entry["total_channels"] = 0
                    entry["mean_thumbnail"] = None
                    entry["stats"] = {}
                combined_blocks[key] = entry

        elif btype == "C2PSA":
            # cv1, m.0 (PSABlock, atomic), cv2
            sub_entries_c2psa = [
                ("cv1", "Conv"),
                ("m.0", "PSABlock"),
                ("cv2", "Conv"),
            ]
            for path, module_type in sub_entries_c2psa:
                key = f"{idx}.{path}"
                t = sub_act.get(key)
                entry = {
                    "type": module_type,
                    "parent": idx,
                    "parent_type": "C2PSA",
                    "path": path,
                }
                if t is not None and torch.is_tensor(t) and t.dim() == 4:
                    entry["shape"] = list(t.shape)
                    entry.update(_render_tensor(t))
                else:
                    entry["shape"] = None
                    entry["channel_pngs"] = []
                    entry["channel_indices"] = []
                    entry["total_channels"] = 0
                    entry["mean_thumbnail"] = None
                    entry["stats"] = {}
                combined_blocks[key] = entry

        elif btype == "Detect":
            # Per-scale heads — reference L1 scales data instead of re-rendering
            detect_payload = l1_data["blocks"].get(idx) or l1_data["blocks"].get(str(idx)) or {}
            l1_scales = detect_payload.get("scales", {})
            scale_map = [
                ("head_p3", "P3"),
                ("head_p4", "P4"),
                ("head_p5", "P5"),
            ]
            for path, scale_key in scale_map:
                key = f"{idx}.{path}"
                scale_data = l1_scales.get(scale_key, {})
                entry = {
                    "type": "DetectHead",
                    "parent": idx,
                    "parent_type": "Detect",
                    "path": path,
                    "scale": scale_key,
                    "classes": scale_data.get("classes", []),
                }
                combined_blocks[key] = entry

    # Step 6: Assemble final data dict
    meta = dict(l1_data["meta"])
    meta["level"] = 2

    return {
        "meta": meta,
        "blocks": combined_blocks,
    }
