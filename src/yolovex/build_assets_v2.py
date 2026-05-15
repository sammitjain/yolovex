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

from .build_assets import CHANNEL_MAX_DIM, TOP_K_CHANNELS, _heatmap_to_b64
from .build_assets_l2 import _preprocess_for_raw_forward
from .block_spec import _trace_with_patches
from .model import DEFAULT_WEIGHTS, get_blocks, load_model


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
    nodes_out: dict[str, dict] = {}

    print(f"  [v2] re-interpreting each block's fx graph to harvest sub-activations...")
    for idx, block in enumerate(blocks):
        cls = type(block).__name__
        key = str(idx)
        entry: dict[str, Any] = {"type": cls, "output": None, "sub": {}}

        if cls == "Detect":
            skipped.append(idx)
            nodes_out[key] = entry
            print(f"    [{idx:>2}] {cls:<10} skipped (deferred)")
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
