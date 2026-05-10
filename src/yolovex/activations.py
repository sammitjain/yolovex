"""Forward-hook based activation capture.

One inference pass populates a dict keyed by block index. Use as a context
manager to ensure hooks are removed even if inference raises.
"""

from __future__ import annotations

from typing import Any

from .model import get_blocks


class ActivationStore:
    """Registers a forward hook on every block and stores its output by index."""

    def __init__(self) -> None:
        self.activations: dict[int, Any] = {}
        self._handles: list = []

    def _make_hook(self, idx: int):
        # Factory binds `idx` per-hook — avoids the closure-over-loop-variable bug.
        def hook(module, inp, out):
            self.activations[idx] = out

        return hook

    def attach(self, blocks: list) -> "ActivationStore":
        self._handles = [
            b.register_forward_hook(self._make_hook(i)) for i, b in enumerate(blocks)
        ]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ActivationStore":
        return self

    def __exit__(self, *exc) -> None:
        self.detach()


def capture(yolo, image_path, imgsz: int = 640) -> tuple[dict[int, Any], list]:
    """Run inference once and return (activations dict, blocks list)."""
    blocks = get_blocks(yolo)
    with ActivationStore().attach(blocks) as store:
        yolo(str(image_path), imgsz=imgsz, verbose=False)
    return store.activations, blocks


def capture_with_results(
    yolo, image_path, imgsz: int = 640
) -> tuple[Any, dict[int, Any], list]:
    """Same as capture(), but also returns the high-level Ultralytics Results object
    (boxes scaled to original image coords, names, etc.)."""
    blocks = get_blocks(yolo)
    with ActivationStore().attach(blocks) as store:
        results = yolo(str(image_path), imgsz=imgsz, verbose=False)
    return results, store.activations, blocks
