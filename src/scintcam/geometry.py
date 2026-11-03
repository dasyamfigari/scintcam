"""Rectification of a square target: radial distortion (k1) + homography, applied as ONE precomputed remap.

The previous approach undistorted each frame (remap) and then applied the homography (warpPerspective): two
interpolations per frame. Here both are composed once into a pair of lookup maps, so each frame is interpolated a
single time, which is both faster and slightly sharper.

Points file schema is the one used by beamview.write_rectification / write_metadata:
  {"setups": {key: {H, out_wh, mm_per_px, K, dist, image_size}}, "metadata": {key: {video, corners, lines}}}
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("scintcam")


def nominal_K(image_size: Tuple[int, int]) -> np.ndarray:
    """Pinhole matrix with focal length = largest image side, principal point at the centre."""
    w, h = image_size
    f = float(max(w, h))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def estimate_k1(lines: Sequence[Sequence[Sequence[float]]], image_size: Tuple[int, int]):
    """Plumb-line estimate of the radial coefficient k1: physically straight edges must image as straight lines.

    cost(k1) = sum over lines, over points of d_perp^2, with d_perp the distance of each undistorted point to the
    total-least-squares line of its own line. Returns (K, dist) with dist = [k1, 0, 0, 0, 0].
    Note: short lines near the image centre cannot constrain k1 (distortion grows as r^2); use long edges far out.
    """
    K = nominal_K(image_size)
    pts = [np.asarray(L, np.float32).reshape(-1, 1, 2) for L in lines if len(L) >= 3]
    if not pts:
        return K, np.zeros(5)

    def residuals(k1):
        dist = np.array([float(k1), 0, 0, 0, 0])
        res = []
        for P in pts:
            u = cv2.undistortPoints(P, K, dist, P=K).reshape(-1, 2)
            uc = u - u.mean(axis=0)
            _, _, vt = np.linalg.svd(uc, full_matrices=False)
            res.append(uc @ vt[-1])
        return np.concatenate(res)

    try:
        from scipy.optimize import least_squares
        k1 = float(least_squares(lambda p: residuals(p[0]), [0.0], method="lm").x[0])
    except ImportError:   # golden-section search on the same scalar cost
        g = (np.sqrt(5.0) - 1) / 2
        a, b = -1.0, 1.0
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = np.sum(residuals(c) ** 2), np.sum(residuals(d) ** 2)
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = np.sum(residuals(c) ** 2)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = np.sum(residuals(d) ** 2)
        k1 = 0.5 * (a + b)
    return K, np.array([k1, 0, 0, 0, 0], float)


@dataclass
class Geometry:
    corners: List[List[float]]                 # 4 camera-pixel points, mapped in order to TL, TR, BR, BL of the output
    lines: List[List[List[float]]]             # straight-edge points for the radial fit
    image_size: Tuple[int, int]                # camera frame (width, height)
    out_size: int
    margin: int
    target_mm: float
    H: np.ndarray = field(repr=False, default=None)
    K: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    _maps: Any = field(repr=False, default=None)

    @property
    def out_wh(self) -> Tuple[int, int]:
        return (self.out_size, self.out_size)

    @property
    def mm_per_px(self) -> float:
        """mm per rectified pixel = target side [mm] / (out_size - 2 margin) [px]."""
        return self.target_mm / float(self.out_size - 2 * self.margin)

    @classmethod
    def build(cls, corners, lines, image_size, out_size: int, margin: int, target_mm: float, radial: bool) -> "Geometry":
        if len(corners) != 4:
            raise ValueError("need 4 corners")
        if out_size - 2 * margin <= 0:
            raise ValueError("out_size - 2*margin must be positive")
        if not target_mm or target_mm <= 0:
            raise ValueError("rectify.target_mm (side of the square target in mm) is not set")
        lines = [L for L in (lines or []) if len(L) >= 3]
        K = dist = None
        c = np.asarray(corners, np.float32)
        if radial and lines:
            K, dist = estimate_k1(lines, image_size)
            log.info("radial distortion k1 = %.4f from %d line(s)", dist[0], len(lines))
            if np.any(dist):
                c = cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2).astype(np.float32)
        S, m = out_size, margin
        dst = np.array([[m, m], [S - m, m], [S - m, S - m], [m, S - m]], np.float32)
        H = cv2.getPerspectiveTransform(c, dst)
        g = cls([list(map(float, p)) for p in corners], [[list(map(float, p)) for p in L] for L in lines],
                tuple(image_size), int(out_size), int(margin), float(target_mm), H, K, dist)
        g._build_maps()
        return g

    def _build_maps(self) -> None:
        """Output pixel q -> undistorted camera point Hinv q -> distorted camera point (through the undistort map)."""
        S = self.out_size
        xs, ys = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
        src = cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(S, S, 2)
        mx, my = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)
        if self.dist is not None and np.any(self.dist):
            ux, uy = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, self.image_size, cv2.CV_32FC1)
            mx, my = (cv2.remap(ux, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1),
                      cv2.remap(uy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1))
        self._maps = cv2.convertMaps(mx, my, cv2.CV_16SC2)   # fixed-point maps: fastest remap

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def as_dict(self) -> Dict[str, Any]:
        return {"corners": self.corners, "corners_map_to": "TL, TR, BR, BL of the rectified image, in this order",
                "lines": self.lines, "H": self.H.tolist(), "K": None if self.K is None else self.K.tolist(),
                "dist": None if self.dist is None else self.dist.tolist(), "image_size": list(self.image_size),
                "out_wh": list(self.out_wh), "mm_per_px": self.mm_per_px, "out_size": self.out_size,
                "margin": self.margin, "target_mm": self.target_mm}


def select_channel(bgr: np.ndarray, channel: str) -> np.ndarray:
    if channel == "r":
        return bgr[:, :, 2]
    if channel == "g":
        return bgr[:, :, 1]
    if channel == "b":
        return bgr[:, :, 0]
    if channel == "luma":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if channel == "r_minus_g":
        return cv2.subtract(bgr[:, :, 2], bgr[:, :, 1])    # saturating uint8 subtraction, clipped at 0
    raise ValueError(f"unknown channel {channel!r} (r | g | b | luma | r_minus_g)")


# --------------------------------------------------------------------------------------------------------------
# Points file (beamview-compatible)
# --------------------------------------------------------------------------------------------------------------
def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        log.warning("could not read %s; starting a new one", path)
        return {}


def _save(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_geometry(path: str, key: str, geom: Geometry, video_path: str) -> None:
    data = _load(path)
    data.setdefault("setups", {})[key] = {"H": geom.H.tolist(), "out_wh": list(geom.out_wh),
                                          "mm_per_px": geom.mm_per_px,
                                          **({"K": geom.K.tolist(), "dist": geom.dist.tolist()} if geom.K is not None else {}),
                                          "image_size": list(geom.image_size)}
    data.setdefault("metadata", {})[key] = {"video": video_path, "corners": geom.corners, "lines": geom.lines}
    _save(path, data)


def load_points(path: str, key: str):
    """Stored (corners, lines) for `key`, or (None, None)."""
    meta = _load(path).get("metadata", {}).get(key)
    if meta and len(meta.get("corners", [])) == 4:
        return meta["corners"], meta.get("lines", [])
    return None, None
