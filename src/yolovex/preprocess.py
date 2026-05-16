"""Image preprocessing for the unfused raw forward path.

Ultralytics' high-level predict path fuses Conv+BN before running, which
collapses the BatchNorm modules and prevents per-block forward hooks from
firing on them. We sidestep that by calling `yolo.model(tensor)` directly —
which requires us to do the letterbox + tensor conversion ourselves.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def _preprocess_for_raw_forward(image_path, imgsz: int) -> torch.Tensor:
    """Letterbox an image and convert to a (1, 3, H, W) float tensor in [0, 1]."""
    from ultralytics.data.augment import LetterBox

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)
    img_lb = lb(image=img)
    return torch.from_numpy(np.ascontiguousarray(img_lb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
