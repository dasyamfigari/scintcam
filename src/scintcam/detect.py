"""Automatic detection of a square target's corners.

The corners are not searched directly: a vertex can be chipped or rounded and sits next to mounting holes. Instead a
straight line is fitted to each EDGE (hundreds of sub-pixel edge points) and the corners are the intersections of
adjacent lines, so they are defined geometrically even when the physical vertex is damaged.
Do it with the beam off and the room lit: with beam, only the irradiated spot glows.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("scintcam")


def contrast_image(img: np.ndarray, mode: str) -> np.ndarray:
    """Image in which the target is bright. r-g: red target (e.g. Cr-doped alumina) on grey/black surroundings;
    luma / inv_luma: bright / dark target. (Saturation does not work: black foam is as saturated as a red plate.)"""
    f = img.astype(np.float32)
    if mode == "r-g":
        c = np.clip(f[:, :, 2] - f[:, :, 1], 0, None)
    elif mode in ("luma", "inv_luma"):
        c = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
        if mode == "inv_luma":
            c = 255.0 - c
    else:
        raise ValueError(f"unknown detect_contrast {mode!r} (r-g | luma | inv_luma)")
    return cv2.GaussianBlur(c, (5, 5), 0)


def _intersect(l1, l2) -> Optional[np.ndarray]:
    """Intersection of two lines given as (x0, y0, vx, vy)."""
    x1, y1, u1, v1 = l1
    x2, y2, u2, v2 = l2
    A = np.array([[u1, -u2], [v1, -v2]], float)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1], float))
    return np.array([x1 + t[0] * u1, y1 + t[0] * v1])


def _edge_points(c: np.ndarray, p: np.ndarray, q: np.ndarray, trim: float) -> np.ndarray:
    """Sub-pixel edge points along side p->q: steepest point of profiles perpendicular to the side, refined with a
    parabola through the three samples around the gradient peak. The ends (trim fraction) are skipped."""
    L = float(np.linalg.norm(q - p))
    d = (q - p) / L
    n = np.array([-d[1], d[0]])
    band = max(4.0, 0.04 * L)
    offs = np.arange(-band, band + 1e-9, 0.25)
    pts = []
    for s in np.linspace(trim * L, (1 - trim) * L, int(min(max(L / 3, 20), 80))):
        c0 = p + d * s
        xy = c0[None, :] + offs[:, None] * n[None, :]
        prof = cv2.remap(c, xy[:, 0].astype(np.float32)[None], xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR).ravel()
        g = np.abs(np.gradient(prof))
        k = int(np.argmax(g))
        if g[k] <= 0:
            continue
        kf = float(k)
        if 0 < k < len(g) - 1:
            den = g[k - 1] - 2 * g[k] + g[k + 1]
            if den != 0:
                kf = k + 0.5 * (g[k - 1] - g[k + 1]) / den
        pts.append(c0 + np.interp(kf, np.arange(len(offs)), offs) * n)
    return np.array(pts, np.float32)


def detect_target(img: np.ndarray, contrast: str = "r-g", trim: float = 0.18):
    """Return (corners 4x2 clockwise on screen, edge points per side, edge-fit rms per side [px]) or None.

    Every rejection is logged with its reason, so a bad detection never silently replaces good corners.
    """
    c = contrast_image(img, contrast)
    if c.max() <= 0:
        log.warning("auto-detect: no contrast in the frame")
        return None
    c8 = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # fill engraved marks
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    if not cs:
        log.warning("auto-detect: no candidate region")
        return None
    cnt = max(cs, key=cv2.contourArea)
    frac = cv2.contourArea(cnt) / float(m.size)
    if not 0.01 < frac < 0.8:
        log.warning("auto-detect: largest region covers %.1f%% of the frame, not a target", 100 * frac)
        return None
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.12, 40):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(float)
            break
    if quad is None:
        log.warning("auto-detect: region is not a quadrilateral")
        return None

    fits = edges = None
    for _ in range(2):        # second pass searches around the refined corners
        fits, edges = [], []
        for i in range(4):
            pts = _edge_points(c, quad[i], quad[(i + 1) % 4], trim)
            if len(pts) < 6:
                log.warning("auto-detect: too few edge points on side %d", i + 1)
                return None
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            fits.append((x0, y0, vx, vy))
            edges.append(pts)
        new = [_intersect(fits[i - 1], fits[i]) for i in range(4)]
        if any(p is None for p in new):
            log.warning("auto-detect: parallel edges")
            return None
        quad = np.array(new)

    rms = [float(np.sqrt(np.mean(((P[:, 0] - x0) * vy - (P[:, 1] - y0) * vx) ** 2)))
           for (x0, y0, vx, vy), P in zip(fits, edges)]
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    h, w = img.shape[:2]
    if max(sides[0], sides[2]) > 2 * min(sides[0], sides[2]) or max(sides[1], sides[3]) > 2 * min(sides[1], sides[3]):
        log.warning("auto-detect: implausible shape (sides %s px)", ", ".join(f"{s:.0f}" for s in sides))
        return None
    if np.any(quad < -2) or np.any(quad[:, 0] > w + 2) or np.any(quad[:, 1] > h + 2):
        log.warning("auto-detect: a corner falls outside the frame")
        return None
    if max(rms) > 3.0:
        log.warning("auto-detect: ragged edges (rms %s px); is the beam on or an edge covered?",
                    ", ".join(f"{x:.2f}" for x in rms))
        return None
    # A target FILLS its quadrilateral (0.985 measured on a chipped plate); a round glow cannot (a disc fills at most
    # pi/4 of an enclosing square; fitted quads measured 0.82-0.90).
    inside = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(inside, [np.round(quad).astype(np.int32)], 255)
    fill = float(((m > 0) & (inside > 0)).sum()) / max(int((inside > 0).sum()), 1)
    if fill < 0.95:
        log.warning("auto-detect: region fills only %.0f%% of its quadrilateral; looks like a beam spot "
                    "(detect with the beam off)", 100 * fill)
        return None
    xs, ys = quad[:, 0], quad[:, 1]
    if np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys) < 0:     # make it clockwise on screen (y down)
        quad, edges, rms = quad[::-1].copy(), edges[::-1], rms[::-1]
    return quad, edges, rms


def order_corners(quad: np.ndarray, reference: Optional[Sequence] = None, first: str = "TL") -> np.ndarray:
    """Put detected corners in the user's order: matched to the corners already placed (any of the 8 cyclic orders,
    so the user's own convention is kept), or, with none placed, starting at `first` and going clockwise."""
    quad = np.asarray(quad, float)
    if reference is not None and len(reference) > 0:
        ref = np.asarray(reference, float)
        cands = [np.roll(b, -s, axis=0) for b in (quad, quad[::-1]) for s in range(4)]
        return min(cands, key=lambda q: float(np.sum(np.linalg.norm(q[:len(ref)] - ref, axis=1))))
    x, y = quad[:, 0], quad[:, 1]
    idx = {"TL": np.argmin(x + y), "BR": np.argmax(x + y), "TR": np.argmax(x - y), "BL": np.argmin(x - y)}
    return np.roll(quad, -int(idx.get(first.upper(), idx["TL"])), axis=0)
