"""Model loading. Thin wrapper over Ultralytics so callers don't import it directly."""

from __future__ import annotations

from ultralytics import YOLO

DEFAULT_WEIGHTS = "yolo26n.pt"


def load_model(weights: str = DEFAULT_WEIGHTS) -> YOLO:
    """Load a YOLO model in eval mode. Weights are downloaded on first use."""
    yolo = YOLO(weights)
    yolo.model.eval()
    return yolo


def get_blocks(yolo: YOLO) -> list:
    """Return the indexed list of blocks (backbone + neck + head)."""
    return list(yolo.model.model)
