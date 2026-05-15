"""Command-line interface for yolovex.

Commands:
- `yolovex layers`   : enriched block table (type, role, shape, params, description)
- `yolovex describe` : full info for a single layer
- `yolovex show`     : render activations to PNG; interactive picker if no layer given
- `yolovex predict`  : detection-head viz — boxes + per-scale and per-class heatmaps
- `yolovex trace`    : follow a single (x,y) location through every layer
- `yolovex report`   : generate a self-contained HTML report
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import torch
import typer

from .activations import capture, capture_with_results
from .graph import write_svg
from .head import render_predictions
from .layers import LayerInfo, build_layer_table
from .model import DEFAULT_WEIGHTS, load_model
from .report import generate_report
from .topology import extract_topology, topology_summary
from .trace import render_trace
from .viz import render_channel, render_mean, render_top_k

app = typer.Typer(help="yolovex — interactive YOLO architecture explorer", no_args_is_help=True)

DEFAULT_IMAGE = Path("assets/sammit_lighthouse.jpg")
DEFAULT_OUT = Path("out")


def _need(image: Path) -> None:
    if not image.exists():
        typer.echo(f"image not found: {image}", err=True)
        raise typer.Exit(code=2)


def _load_and_capture(image: Path, imgsz: int, weights: str):
    _need(image)
    yolo = load_model(weights)
    activations, blocks = capture(yolo, image, imgsz=imgsz)
    return blocks, activations


def _load_capture_full(image: Path, imgsz: int, weights: str):
    _need(image)
    yolo = load_model(weights)
    results, activations, blocks = capture_with_results(yolo, image, imgsz=imgsz)
    return yolo, results, blocks, activations


def _print_table(rows: list[LayerInfo]) -> None:
    header = f"{'idx':>3}  {'type':<10}  {'role':<7}  {'output shape':<22}  {'params':>10}  description"
    typer.echo(header)
    typer.echo("-" * len(header))
    for r in rows:
        shape = r.output_shape if len(r.output_shape) <= 22 else r.output_shape[:19] + "..."
        typer.echo(
            f"{r.index:>3}  {r.type_name:<10}  {r.role:<7}  {shape:<22}  {r.n_params:>10,}  {r.description}"
        )


def _validate_idx(idx: int, n: int) -> None:
    if not (0 <= idx < n):
        typer.echo(f"layer index {idx} out of range; valid: 0..{n - 1}", err=True)
        raise typer.Exit(code=1)


# ---------- commands ----------

@app.command()
def layers(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
):
    """Print an enriched table of every block in the model."""
    blocks, activations = _load_and_capture(image, imgsz, weights)
    _print_table(build_layer_table(blocks, activations))


@app.command()
def describe(
    layer_idx: int = typer.Argument(...),
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
):
    """Dump everything we know about a single layer."""
    blocks, activations = _load_and_capture(image, imgsz, weights)
    _validate_idx(layer_idx, len(blocks))
    info = build_layer_table(blocks, activations)[layer_idx]
    out = activations.get(layer_idx)

    typer.echo(f"Layer {info.index}: {info.type_name}  [{info.role}]")
    typer.echo(f"  description : {info.description}")
    typer.echo(f"  output shape: {info.output_shape}")
    typer.echo(f"  parameters  : {info.n_params:,}")
    if torch.is_tensor(out) and out.dim() == 4:
        t = out.detach().float().cpu()[0]
        typer.echo(f"  channels    : {t.shape[0]}")
        typer.echo(f"  spatial     : {t.shape[1]} × {t.shape[2]}")
        typer.echo(
            f"  activation  : min={t.min().item():.3f}  max={t.max().item():.3f}  "
            f"mean={t.mean().item():.3f}  std={t.std().item():.3f}"
        )
        scores = t.abs().mean(dim=(1, 2))
        top = torch.argsort(scores, descending=True)[:5].tolist()
        typer.echo(f"  top channels: {top}  (by mean |activation|)")
    typer.echo("")
    typer.echo("module repr:")
    typer.echo(textwrap.indent(str(blocks[layer_idx]), "  "))


@app.command()
def show(
    layer_idx: Optional[int] = typer.Argument(None),
    mean: bool = typer.Option(False, "--mean"),
    channel: Optional[int] = typer.Option(None, "--channel", "-c"),
    top: Optional[int] = typer.Option(None, "--top", "-t"),
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out_dir: Path = typer.Option(DEFAULT_OUT, "--out"),
):
    """Render a feature-block's activations to PNG.

    For the detection head, use `yolovex predict` instead.
    """
    blocks, activations = _load_and_capture(image, imgsz, weights)
    rows = build_layer_table(blocks, activations)

    if layer_idx is None:
        _print_table(rows)
        typer.echo("")
        layer_idx = typer.prompt(f"Pick a layer [0-{len(blocks) - 1}]", type=int)
    _validate_idx(layer_idx, len(blocks))

    out = activations.get(layer_idx)
    info = rows[layer_idx]
    if not (torch.is_tensor(out) and out.dim() == 4):
        typer.echo(
            f"layer {layer_idx} ({info.type_name}) doesn't return a (B,C,H,W) tensor "
            f"— got {type(out).__name__}; try `yolovex predict` for the detection head",
            err=True,
        )
        raise typer.Exit(code=1)

    n_set = sum([mean, channel is not None, top is not None])
    if n_set > 1:
        typer.echo("specify at most one of --mean / --channel / --top", err=True)
        raise typer.Exit(code=1)
    if n_set == 0:
        choice = typer.prompt("View: (m)ean / (c)hannel / (t)op-K", default="m").strip().lower()
        if choice.startswith("m"):
            mean = True
        elif choice.startswith("c"):
            channel = typer.prompt("Channel index", type=int)
        elif choice.startswith("t"):
            top = typer.prompt("Top how many?", default=16, type=int)
        else:
            typer.echo(f"unknown choice: {choice!r}", err=True)
            raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"layer{layer_idx:02d}_{info.type_name}"
    if mean:
        path = render_mean(out, layer_idx, info.type_name, out_dir / f"{stem}_mean.png")
    elif channel is not None:
        path = render_channel(out, layer_idx, info.type_name, channel,
                              out_dir / f"{stem}_c{channel}.png")
    else:
        path = render_top_k(out, layer_idx, info.type_name, top, out_dir / f"{stem}_top{top}.png")
    typer.echo(f"saved {path}")


@app.command()
def predict(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    conf: float = typer.Option(0.25, "--conf", help="confidence threshold for boxes/top-classes"),
    top_classes: int = typer.Option(4, "--top-classes", help="how many detected classes to render heatmaps for"),
    out_dir: Path = typer.Option(DEFAULT_OUT, "--out"),
):
    """Detection-head viz: boxes + per-scale max-class heatmaps + per-class heatmaps."""
    yolo, results, blocks, activations = _load_capture_full(image, imgsz, weights)
    head_idx = len(blocks) - 1
    head_out = activations[head_idx]
    if not (isinstance(head_out, tuple) and isinstance(head_out[1], dict)):
        typer.echo(f"unexpected head output type: {type(head_out).__name__}", err=True)
        raise typer.Exit(code=1)

    out_path = out_dir / f"predict_{image.stem}.png"
    render_predictions(
        results, head_out, image, yolo.names, out_path,
        conf_thresh=conf, top_classes=top_classes,
    )
    typer.echo(f"saved {out_path}")


@app.command()
def trace(
    x: int = typer.Argument(..., help="x pixel coordinate in the input image"),
    y: int = typer.Argument(..., help="y pixel coordinate in the input image"),
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out_dir: Path = typer.Option(DEFAULT_OUT, "--out"),
):
    """Trace a single (x, y) location through every layer."""
    blocks, activations = _load_and_capture(image, imgsz, weights)
    out_path = out_dir / f"trace_{image.stem}_x{x}_y{y}.png"
    try:
        render_trace(activations, blocks, image, x, y, out_path)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"saved {out_path}")


@app.command()
def report(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out: Path = typer.Option(DEFAULT_OUT / "report.html", "--out"),
):
    """Generate a self-contained HTML report (no JS, all images embedded)."""
    yolo, results, blocks, activations = _load_capture_full(image, imgsz, weights)
    generate_report(results, activations, blocks, image, weights, out)
    typer.echo(f"saved {out}")


@app.command("build-assets")
def build_assets_cmd(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out: Path = typer.Option(Path("frontend/data.js"), "--out"),
):
    """Render activations + predictions to data.js consumed by the frontend."""
    from .build_assets import build, write_data_js

    if not image.exists():
        typer.echo(f"image not found: {image}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"running inference on {image}...")
    data = build(image, weights=weights, imgsz=imgsz)
    write_data_js(data, out)
    n_blocks = len(data["blocks"])
    n_pngs = sum(len(b.get("channel_pngs", [])) for b in data["blocks"].values())
    n_classes = sum(
        len(s.get("classes", []))
        for b in data["blocks"].values()
        for s in (b.get("scales") or {}).values()
    )
    typer.echo(
        f"saved {out} — {n_blocks} blocks, {n_pngs} channel PNGs, "
        f"{n_classes} class PNGs, {out.stat().st_size // 1024} KB"
    )


@app.command("build-assets-l2")
def build_assets_l2_cmd(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out: Path = typer.Option(Path("frontend/data-l2.js"), "--out"),
):
    """Render sub-module activations to data-l2.js consumed by the L2 frontend page."""
    from .build_assets_l2 import build as l2_build
    from .build_assets import write_data_js

    if not image.exists():
        typer.echo(f"image not found: {image}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"running L2 inference on {image}...")
    data = l2_build(image, weights=weights, imgsz=imgsz)
    write_data_js(data, out)
    n_blocks = len(data["blocks"])
    n_pngs = sum(len(b.get("channel_pngs", [])) for b in data["blocks"].values())
    typer.echo(
        f"saved {out} — {n_blocks} entries (parent blocks + sub-nodes), "
        f"{n_pngs} channel PNGs, {out.stat().st_size // 1024} KB"
    )


@app.command("build-assets-v2")
def build_assets_v2_cmd(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out: Path = typer.Option(Path("frontend/activations-v2.js"), "--out"),
):
    """Render per-fx-node activations to activations-v2.js for the v2 frontend."""
    from .build_assets_v2 import build as v2_build, write_activations_js

    if not image.exists():
        typer.echo(f"image not found: {image}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"running v2 activation capture on {image}...")
    data = v2_build(image, weights=weights, imgsz=imgsz)
    write_activations_js(data, out)
    n_blocks = len(data["nodes"])
    n_subs = sum(len(b.get("sub", {})) for b in data["nodes"].values())
    typer.echo(
        f"saved {out} — {n_blocks} blocks, {n_subs} sub-nodes, "
        f"skipped={data['meta']['skipped']}, {out.stat().st_size // 1024} KB"
    )


@app.command()
def graph(
    image: Path = typer.Option(DEFAULT_IMAGE, "--image", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    weights: str = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),
    out: Path = typer.Option(DEFAULT_OUT / "architecture.svg", "--out"),
    print_only: bool = typer.Option(False, "--print-only", help="just print the topology, don't render SVG"),
):
    """Render the architecture as an SVG (topology from model, layout per paper)."""
    blocks, activations = _load_and_capture(image, imgsz, weights)
    nodes, edges = extract_topology(blocks, activations)
    typer.echo(topology_summary(nodes))
    if print_only:
        return
    write_svg(nodes, edges, out)
    typer.echo("")
    typer.echo(f"saved {out}")


if __name__ == "__main__":
    app()
