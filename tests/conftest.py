import time

import cv2
import numpy as np
import pytest

from scintcam.config import load_config


def plate_scene(w=640, h=480, corners=((150, 90), (470, 110), (450, 400), (130, 380)), side=600, beam=None):
    """Synthetic camera frame: grey background, red square plate seen in perspective, optional Gaussian beam spot."""
    plate = np.zeros((side, side, 3), np.uint8)
    plate[:] = (40, 40, 170)                                   # BGR red plate
    if beam is not None:
        yy, xx = np.mgrid[0:side, 0:side]
        g = beam * np.exp(-(((xx - side / 2) ** 2) + ((yy - side / 2) ** 2)) / (2 * (side / 10) ** 2))
        plate[:, :, 2] = np.clip(plate[:, :, 2] + g, 0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    H = cv2.getPerspectiveTransform(src, np.float32(corners))
    bg = np.full((h, w, 3), 120, np.uint8)
    warped = cv2.warpPerspective(plate, H, (w, h))
    mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), H, (w, h))
    bg[mask > 127] = warped[mask > 127]
    return bg


class FakeCap:
    """Stands in for cv2.VideoCapture: fixed frame, optional frame period, driver timestamps."""

    def __init__(self, frame, period=0.0):
        self.frame, self.period, self.n, self.t0 = frame, period, 0, time.monotonic()
        self.props = {}

    def read(self):
        if self.period:
            time.sleep(self.period)
        self.n += 1
        return True, self.frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 1000.0 * (time.monotonic() - self.t0)
        return float(self.props.get(prop, 0.0))

    def set(self, prop, val):
        self.props[prop] = val
        return True

    def isOpened(self):
        return True

    def release(self):
        pass


@pytest.fixture
def cfg(tmp_path):
    c = load_config(None)
    c["rectify"]["target_mm"] = 32.5
    c["rectify"]["points_json"] = str(tmp_path / "points.json")
    c["recording"]["dir"] = str(tmp_path / "rec")
    c["camera"]["use_v4l2ctl"] = False
    return c
