"""Export C2PSA attention for the frontend attention prototype.

The prototype wants the post-softmax attention tensor in the browser, not a
pre-rendered GIF. This script reuses the capture hook from scripts/viz_attention.py
and writes a compact uint8 payload that can be loaded as either JSON or a
static-friendly JS global.

Usage:
  uv run python scripts/export_attention_json.py
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from viz_attention import capture_attention  # noqa: E402
from yolovex.model import DEFAULT_WEIGHTS  # noqa: E402


def _image_data_url(image_bgr: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("could not JPEG-encode attention preview image")
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _quantize_heads(attn: np.ndarray) -> dict[str, Any]:
    """Quantize [heads, N, N] float attention to a head-major uint8 blob."""
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


def build_payload(image: Path, imgsz: int, weights: str, jpeg_quality: int) -> dict[str, Any]:
    attn, grid_h, grid_w, image_bgr = capture_attention(image, imgsz, weights)
    attn_np = attn[0].numpy().astype(np.float32)
    heads, queries, keys = attn_np.shape
    expected = grid_h * grid_w
    if queries != expected or keys != expected:
        raise RuntimeError(
            f"attention shape {attn_np.shape} does not match grid {grid_h}x{grid_w}"
        )

    image_h, image_w = image_bgr.shape[:2]
    return {
        "version": 1,
        "meta": {
            "kind": "attention-prototype",
            "source_image": str(image),
            "weights": weights,
            "imgsz": imgsz,
            "image_w": image_w,
            "image_h": image_h,
            "grid_w": grid_w,
            "grid_h": grid_h,
            "heads": heads,
            "tokens": expected,
        },
        "image": {
            "mime": "image/jpeg",
            "data_url": _image_data_url(image_bgr, jpeg_quality),
        },
        "attention": _quantize_heads(attn_np),
    }


def write_payload(payload: dict[str, Any], out: Path, global_name: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    if out.suffix == ".js":
        out.write_text(f"window.{global_name} = {text};\n", encoding="utf-8")
    else:
        out.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("assets/sammit_lighthouse.jpg"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=Path("frontend/attention-prototype-data.js"))
    parser.add_argument("--global-name", type=str, default="YV_ATTENTION_PROTOTYPE")
    parser.add_argument("--jpeg-quality", type=int, default=84)
    args = parser.parse_args()

    payload = build_payload(args.image, args.imgsz, args.weights, args.jpeg_quality)
    write_payload(payload, args.out, args.global_name)

    shape = payload["attention"]["shape"]
    print(
        f"wrote {args.out} with attention shape={shape} "
        f"and grid={payload['meta']['grid_h']}x{payload['meta']['grid_w']}"
    )


if __name__ == "__main__":
    main()
