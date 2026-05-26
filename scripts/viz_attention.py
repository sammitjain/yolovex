"""Visualize C2PSA self-attention as a per-query-pixel animation.

For each query position (i, j) in the C2PSA attention map (20x20 for yolo26n
at imgsz=640), we have a 20x20 vector of attention weights — "how much does
this query attend to every other position?". We upsample those weights back
to the input resolution, overlay them on the input image, and animate the
2D traversal so a viewer can watch which regions light up together.

Why it's interesting: pure self-attention is permutation-invariant in
theory, but C2PSA's Attention adds a depthwise 3x3 position bias `pe` —
combined with the backbone's learned semantics, query pixels in the
"object region" tend to attend strongly to *other* pixels of the same
object. On a photo of a car, the wheels often light up together; on a
person, head and feet co-activate.

Usage:
  uv run python scripts/viz_attention.py \\
      --image assets/sammit_lighthouse.jpg \\
      --out out/attention.gif \\
      --head mean \\
      --order raster

  # Only render a sparse subset of queries (every Nth) for a quicker test
  uv run python scripts/viz_attention.py --stride 4 --fps 6

  # Write individual frames to a directory (useful for previewing in browser)
  uv run python scripts/viz_attention.py --frames-dir out/attention_frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

# Allow running directly from repo root: `uv run python scripts/viz_attention.py`
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yolovex.model import DEFAULT_WEIGHTS, get_blocks, load_model
from yolovex.preprocess import _preprocess_for_raw_forward


# ---------------------------------------------------------------------------
# Hook: capture the softmaxed attention tensor from C2PSA's Attention module
# ---------------------------------------------------------------------------

def attach_attn_hook(attn_module) -> dict:
    """Monkey-patch attn_module.forward to stash the post-softmax attn tensor.

    Returns a dict that will be populated with {'attn': Tensor[B, heads, N, N],
    'H': int, 'W': int} after the forward pass runs.
    """
    captured: dict = {}
    original_forward = attn_module.forward

    def patched_forward(x):
        B, C, H, W = x.shape
        N = H * W
        qkv = attn_module.qkv(x)
        q, k, v = qkv.view(
            B, attn_module.num_heads,
            attn_module.key_dim * 2 + attn_module.head_dim, N
        ).split(
            [attn_module.key_dim, attn_module.key_dim, attn_module.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * attn_module.scale
        attn = attn.softmax(dim=-1)
        captured["attn"] = attn.detach().cpu()
        captured["H"] = H
        captured["W"] = W
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + attn_module.pe(
            v.reshape(B, C, H, W)
        )
        return attn_module.proj(out)

    attn_module.forward = patched_forward

    def restore():
        attn_module.forward = original_forward

    return captured, restore


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def build_query_order(H: int, W: int, mode: str, stride: int) -> list[tuple[int, int]]:
    """Return list of (i, j) query positions to visit."""
    coords = []
    if mode == "raster":
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                coords.append((i, j))
    elif mode == "boustrophedon":
        for i in range(0, H, stride):
            row = list(range(0, W, stride))
            if (i // stride) % 2 == 1:
                row = list(reversed(row))
            for j in row:
                coords.append((i, j))
    elif mode == "diagonal":
        for d in range(0, H + W, stride):
            for i in range(0, H, stride):
                j = d - i
                if 0 <= j < W:
                    coords.append((i, j))
    else:
        raise ValueError(f"unknown order mode: {mode}")
    return coords


def render_frame(
    image_bgr: np.ndarray,
    attn_2d: np.ndarray,   # shape (H_attn, W_attn) — single query's attention map
    query_xy: tuple[int, int],   # (row, col) in attention grid
    *,
    alpha: float = 0.55,
    cmap_name: str = "inferno",
    marker_radius: int = 6,
    border_color: tuple[int, int, int] = (10, 10, 14),
    marker_color: tuple[int, int, int] = (255, 255, 255),
    norm_lo: float | None = None,    # for "global" normalize: pass the global min/max
    norm_hi: float | None = None,
) -> np.ndarray:
    """Compose one frame: input image + attention heatmap overlay + query marker.

    If `norm_lo`/`norm_hi` are provided, the attention is mapped through that
    fixed range (used for "global" normalisation). Otherwise we stretch each
    query independently to [0, 1].
    """
    H_img, W_img = image_bgr.shape[:2]
    H_attn, W_attn = attn_2d.shape

    a = attn_2d.astype(np.float32)
    if norm_lo is not None and norm_hi is not None and norm_hi > norm_lo:
        a_norm = np.clip((a - norm_lo) / (norm_hi - norm_lo), 0.0, 1.0)
    else:
        a_min, a_max = float(a.min()), float(a.max())
        if a_max > a_min:
            a_norm = (a - a_min) / (a_max - a_min)
        else:
            a_norm = np.zeros_like(a)

    # Upsample to image resolution (bilinear via OpenCV).
    a_up = cv2.resize(a_norm, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    # Apply matplotlib colormap → uint8 RGB.
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    heat_rgba = (cmap(a_up) * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2BGR)

    # Alpha blend over the input image.
    out = cv2.addWeighted(image_bgr, 1 - alpha, heat_bgr, alpha, 0)

    # Map the query (row, col) from attention grid → image pixel coords.
    qi, qj = query_xy
    cy = int((qi + 0.5) * H_img / H_attn)
    cx = int((qj + 0.5) * W_img / W_attn)

    # Cross-hair marker — outline ring + filled dot for contrast on any colour.
    cv2.circle(out, (cx, cy), marker_radius + 2, border_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(out, (cx, cy), marker_radius, marker_color, thickness=-1, lineType=cv2.LINE_AA)

    return out


def annotate_frame(
    frame_bgr: np.ndarray,
    query_xy: tuple[int, int],
    H_attn: int,
    W_attn: int,
    *,
    head_label: str = "mean",
) -> np.ndarray:
    """Stamp the query coordinate + head label in a small chrome strip."""
    H_img, W_img = frame_bgr.shape[:2]
    bar_h = 28
    out = np.zeros((H_img + bar_h, W_img, 3), dtype=np.uint8)
    out[:H_img] = frame_bgr
    out[H_img:] = (18, 22, 30)

    qi, qj = query_xy
    text = f"query ({qi:>2},{qj:>2}) of {H_attn}x{W_attn}    head: {head_label}"
    cv2.putText(
        out, text,
        org=(12, H_img + 18),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.45,
        color=(220, 226, 240),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def capture_attention(image_path: Path, imgsz: int, weights: str):
    """Return (attn_tensor: [B, heads, N, N], H, W, image_bgr_letterboxed)."""
    yolo = load_model(weights)
    blocks = get_blocks(yolo)
    c2psa = blocks[10]
    if type(c2psa).__name__ != "C2PSA":
        raise RuntimeError(f"expected C2PSA at block 10, got {type(c2psa).__name__}")
    attn_module = c2psa.m[0].attn

    captured, restore = attach_attn_hook(attn_module)

    try:
        x = _preprocess_for_raw_forward(image_path, imgsz)
        with torch.no_grad():
            yolo.model(x)
    finally:
        restore()

    if "attn" not in captured:
        raise RuntimeError("attention hook didn't fire — forward pass aborted?")

    # Rehydrate the letterboxed BGR image for overlay rendering. The
    # preprocessor letterboxes + RGB-converts + normalises, so we reverse the
    # last two steps (uint8 BGR for OpenCV draw calls).
    img_rgb = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return captured["attn"], captured["H"], captured["W"], img_bgr


def select_head_slice(attn: torch.Tensor, head: str) -> torch.Tensor:
    """attn: [B, heads, N, N] -> [N, N] for a chosen head ("mean" | "0" | "1" | ...)."""
    a = attn[0]  # drop batch -> [heads, N, N]
    if head == "mean":
        return a.mean(dim=0)
    idx = int(head)
    if not (0 <= idx < a.shape[0]):
        raise ValueError(f"head {idx} out of range (model has {a.shape[0]} heads)")
    return a[idx]


def select_all_head_slices(attn: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Return [(label, [N,N]), ...] for every individual head — used by --head all."""
    a = attn[0]
    return [(f"head {i}", a[i]) for i in range(a.shape[0])]


def hstack_with_gutter(frames: list[np.ndarray], gutter_px: int = 12,
                       gutter_color: tuple[int, int, int] = (18, 22, 30)) -> np.ndarray:
    """Glue multiple frames horizontally with a gutter strip in between."""
    Hs = [f.shape[0] for f in frames]
    H = max(Hs)
    padded = []
    for f in frames:
        if f.shape[0] < H:
            pad = np.zeros((H - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            pad[:] = gutter_color
            f = np.vstack([f, pad])
        padded.append(f)
    out_parts = []
    for i, f in enumerate(padded):
        if i > 0:
            gutter = np.zeros((H, gutter_px, 3), dtype=np.uint8)
            gutter[:] = gutter_color
            out_parts.append(gutter)
        out_parts.append(f)
    return np.hstack(out_parts)


def resize_to_width(frames: list[np.ndarray], max_width: int) -> list[np.ndarray]:
    if max_width <= 0:
        return frames
    h, w = frames[0].shape[:2]
    if w <= max_width:
        return frames
    new_w = max_width
    new_h = int(round(h * (new_w / w)))
    # Even dims play nicer with mp4 encoders.
    new_h += new_h % 2
    new_w += new_w % 2
    return [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]


def save_gif(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Quantise to a palette so file size stays sane (default Pillow GIF
    # encode without quantization produces enormous files for photo-heavy
    # frames). 128 colors keeps the heatmap looking smooth.
    pil_frames = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        pil_frames.append(pil)
    duration_ms = max(int(round(1000 / max(fps, 1))), 20)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def save_mp4(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    # mp4v works without an ffmpeg binary on macOS. H264 codec ('avc1') is
    # smaller but needs ffmpeg / codec packs that aren't guaranteed in uv envs.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")
    for f in frames:
        writer.write(f)
    writer.release()


def save_frames(frames: list[np.ndarray], frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(frames))))
    for i, f in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"frame_{i:0{width}d}.png"), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=Path("assets/sammit_lighthouse.jpg"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=Path, default=Path("out/attention.gif"))
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="also write individual frames here (useful for preview)")
    p.add_argument("--head", type=str, default="mean",
                   help='"mean" (default) | head index ("0", "1", ...) | "all" (side-by-side per head)')
    p.add_argument("--normalize", choices=["per-query", "global"], default="per-query",
                   help='"per-query" (default — every frame stretched to [0,1]) | "global" (one fixed range across the whole animation, more honest about magnitude)')
    p.add_argument("--order", choices=["raster", "boustrophedon", "diagonal"],
                   default="boustrophedon")
    p.add_argument("--stride", type=int, default=1,
                   help="step between consecutive query positions (1 = every pixel)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.55,
                   help="heatmap overlay opacity [0..1]")
    p.add_argument("--cmap", type=str, default="inferno",
                   help="matplotlib colormap name (e.g. inferno, magma, viridis, turbo)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap on number of frames rendered (for quick previews)")
    p.add_argument("--max-width", type=int, default=360,
                   help="downsize output frames to this pixel width (0 = no resize). "
                        "Smaller = faster encoding + smaller GIF. mp4 output is fine at full res.")
    p.add_argument("--format", choices=["gif", "mp4", "both"], default="gif",
                   help="output format. mp4 is ~20x smaller than gif for the same quality.")
    args = p.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    print(f"loading model and capturing attention for {args.image}...")
    attn, H, W, img_bgr = capture_attention(args.image, args.imgsz, args.weights)
    print(f"  attention tensor: shape={tuple(attn.shape)}  feature map={H}x{W}")

    # Resolve head selection -> a list of (label, [N, N] tensor) panels.
    if args.head == "all":
        panels = select_all_head_slices(attn)
    else:
        panels = [(f"head {args.head}" if args.head != "mean" else "mean",
                   select_head_slice(attn, args.head))]
    panels_np = [(label, t.numpy()) for label, t in panels]

    # Global normalisation reference (only used when --normalize global).
    # We compute one (lo, hi) per panel so all heads share their own scale —
    # comparing magnitudes across heads at the same query stays meaningful.
    if args.normalize == "global":
        norms = [(float(arr.min()), float(arr.max())) for _, arr in panels_np]
    else:
        norms = [(None, None) for _ in panels_np]

    coords = build_query_order(H, W, args.order, args.stride)
    if args.max_frames is not None:
        coords = coords[: args.max_frames]
    print(f"  rendering {len(coords)} frames "
          f"(head={args.head}, order={args.order}, stride={args.stride}, normalize={args.normalize})...")

    frames = []
    for (qi, qj) in coords:
        panel_frames = []
        for (label, head_slice), (lo, hi) in zip(panels_np, norms):
            row = head_slice[qi * W + qj].reshape(H, W)
            frame = render_frame(
                img_bgr, row, (qi, qj),
                alpha=args.alpha, cmap_name=args.cmap,
                norm_lo=lo, norm_hi=hi,
            )
            frame = annotate_frame(frame, (qi, qj), H, W, head_label=label)
            panel_frames.append(frame)
        composed = hstack_with_gutter(panel_frames) if len(panel_frames) > 1 else panel_frames[0]
        frames.append(composed)

    if args.max_width > 0:
        frames = resize_to_width(frames, args.max_width)
        print(f"  resized to {frames[0].shape[1]}x{frames[0].shape[0]}")

    if args.frames_dir is not None:
        save_frames(frames, args.frames_dir)
        print(f"  wrote {len(frames)} frames to {args.frames_dir}/")

    targets = [args.format] if args.format != "both" else ["gif", "mp4"]
    for fmt in targets:
        out_path = args.out.with_suffix(f".{fmt}")
        if fmt == "gif":
            save_gif(frames, out_path, args.fps)
        else:
            save_mp4(frames, out_path, args.fps)
        print(f"  wrote {fmt.upper()} to {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
