"""numpy BGR image -> Tk photo image."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageTk


def to_photo(bgr: np.ndarray, size=None) -> ImageTk.PhotoImage:
    """Optionally resized to size=(w, h): nearest neighbour when enlarging (pixels stay visible), area when shrinking."""
    if size is not None and (size[0] != bgr.shape[1] or size[1] != bgr.shape[0]) and size[0] > 0 and size[1] > 0:
        interp = cv2.INTER_NEAREST if size[0] > bgr.shape[1] else cv2.INTER_AREA
        bgr = cv2.resize(bgr, (int(size[0]), int(size[1])), interpolation=interp)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def fit(w: int, h: int, avail_w: int, avail_h: int) -> tuple:
    """Largest (w', h') with the aspect ratio of (w, h) that fits in (avail_w, avail_h)."""
    s = min(avail_w / float(w), avail_h / float(h))
    return max(1, int(w * s)), max(1, int(h * s))
