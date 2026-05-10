"""Layer metadata: type descriptions, role tagging (feature vs head), and the
combined info table that the CLI renders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Short human-readable description per known YOLO block class. Falls back to "—".
BLOCK_DESCRIPTIONS: dict[str, str] = {
    "Conv": "Conv2d + BatchNorm + activation — basic feature transform",
    "C2f": "cross-stage partial bottleneck (fast) — main feature-mixing unit",
    "C3": "cross-stage partial bottleneck (3-conv variant)",
    "C3k2": "C3 variant used in newer YOLO releases",
    "C2": "lighter cross-stage block",
    "C2PSA": "C2 block augmented with position-sensitive (self-)attention",
    "PSA": "position-sensitive self-attention block",
    "SPPF": "spatial pyramid pooling (fast) — multi-scale context via stacked max-pools",
    "SPP": "spatial pyramid pooling",
    "Concat": "concatenates feature maps along channel dim — joins skip connections",
    "Upsample": "nearest-neighbor upsample — increases spatial resolution",
    "Detect": "detection head — outputs box regressions + class logits per scale",
    "v10Detect": "YOLOv10 detection head",
    "Bottleneck": "residual bottleneck (1x1 → 3x3 → add)",
    "DFL": "distribution focal loss component of the head",
    "Focus": "space-to-depth slicing (legacy YOLO)",
    "Pose": "pose estimation head",
    "Segment": "segmentation head",
    "OBB": "oriented bounding box head",
}

# Class names treated as "the head" for role detection. The first one we see
# marks the boundary between feature blocks and head blocks.
HEAD_CLASSES = {"Detect", "v10Detect", "Pose", "Segment", "OBB"}


def describe_block_type(name: str) -> str:
    return BLOCK_DESCRIPTIONS.get(name, "—")


def find_head_index(blocks: list) -> int | None:
    for i, b in enumerate(blocks):
        if type(b).__name__ in HEAD_CLASSES:
            return i
    return None


def role_of(idx: int, head_idx: int | None) -> str:
    if head_idx is None:
        return "feature"
    return "head" if idx >= head_idx else "feature"


def shape_of(activation: Any) -> str:
    """Best-effort string for the activation shape — works for tensors and tuples."""
    if hasattr(activation, "shape"):
        return str(tuple(activation.shape))
    if isinstance(activation, (tuple, list)):
        return f"<{type(activation).__name__} of {len(activation)}>"
    if activation is None:
        return "—"
    return f"<{type(activation).__name__}>"


@dataclass
class LayerInfo:
    index: int
    type_name: str
    role: str
    description: str
    output_shape: str
    n_params: int


def build_layer_table(blocks: list, captured: dict[int, Any]) -> list[LayerInfo]:
    head_idx = find_head_index(blocks)
    rows: list[LayerInfo] = []
    for i, m in enumerate(blocks):
        type_name = type(m).__name__
        rows.append(
            LayerInfo(
                index=i,
                type_name=type_name,
                role=role_of(i, head_idx),
                description=describe_block_type(type_name),
                output_shape=shape_of(captured.get(i)),
                n_params=sum(p.numel() for p in m.parameters()),
            )
        )
    return rows
