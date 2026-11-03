"""Images for display: colour-mapped view, grid, 1D profile strips and the I_tot seismograph. Pure numpy/OpenCV."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

COLORMAPS = ("hot", "inferno", "magma", "plasma", "viridis", "cividis", "turbo", "jet", "parula", "bone", "gray",
             "rainbow", "ocean", "summer", "spring", "cool", "hsv", "pink", "autumn", "winter", "twilight")


def colormap_lut(name: str) -> np.ndarray:
    """256x1x3 BGR lookup table of an OpenCV colormap ('gray' = identity)."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    if name == "gray":
        return cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR).reshape(256, 1, 3)
    cid = getattr(cv2, "COLORMAP_" + name.upper(), None)
    if cid is None:
        raise ValueError(f"colormap {name!r} not available in this OpenCV build")
    return cv2.applyColorMap(ramp, cid).reshape(256, 1, 3)


def view_image(rect: np.ndarray, ch: np.ndarray, view: str, lut: np.ndarray, sat_level: float,
               sat_color: Sequence[int]) -> np.ndarray:
    if view == "color":
        out = rect.copy()
    else:
        c8 = ch if ch.dtype == np.uint8 else np.clip(ch, 0, 255).astype(np.uint8)
        g = cv2.cvtColor(c8, cv2.COLOR_GRAY2BGR)
        out = g if view == "gray" else cv2.LUT(g, lut)
    if sat_level and view != "color":
        out[ch >= sat_level] = tuple(int(v) for v in sat_color)
    return out


def draw_grid(img: np.ndarray, mm_per_px: float, step_mm: float) -> np.ndarray:
    """Crosshair and grid from the image centre, drawn by inverting pixels (visible on any background)."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    img[cy, :] = 255 - img[cy, :]
    img[:, cx] = 255 - img[:, cx]
    step = step_mm / mm_per_px if step_mm > 0 else 0
    if step >= 6:
        k = 1
        while k * step < max(w, h):
            d = int(round(k * step))
            for x in (cx - d, cx + d):
                if 0 <= x < w:
                    img[::3, x] = 255 - img[::3, x]
            for y in (cy - d, cy + d):
                if 0 <= y < h:
                    img[y, ::3] = 255 - img[y, ::3]
            k += 1
    return img


def _strip(prof: np.ndarray, length: int, thick: int, full: float, mark: Optional[int], vertical: bool) -> np.ndarray:
    panel = np.full((length, thick, 3) if vertical else (thick, length, 3), 18, np.uint8)
    if prof.size < 2:
        return panel
    if prof.size != length:
        prof = np.interp(np.linspace(0, prof.size - 1, length), np.arange(prof.size), prof)
    pad, use = 3, thick - 6
    amp = np.clip(prof / full, 0, 1) * use
    half = int(round(0.5 * use * min(float(prof.max()) / full, 1.0)))
    idx = np.arange(length)
    if vertical:
        pts = np.stack([pad + amp, idx], 1).astype(np.int32)
        cv2.line(panel, (pad + half, 0), (pad + half, length), (140, 70, 70), 1)
        if mark is not None:
            cv2.line(panel, (0, mark), (thick, mark), (90, 90, 90), 1)
    else:
        pts = np.stack([idx, thick - pad - amp], 1).astype(np.int32)
        cv2.line(panel, (0, thick - pad - half), (length, thick - pad - half), (140, 70, 70), 1)
        if mark is not None:
            cv2.line(panel, (mark, 0), (mark, thick), (90, 90, 90), 1)
    cv2.polylines(panel, [pts], False, (80, 220, 80), 1)
    return panel


def with_profiles(img: np.ndarray, px: np.ndarray, py: np.ndarray, size: int, full: float,
                  ipeak: Optional[Tuple[int, int]]) -> np.ndarray:
    """[ image | y profile ] over [ x profile | blank ]. The faint horizontal/vertical line marks half maximum."""
    h, w = img.shape[:2]
    sx = _strip(px, w, size, full, ipeak[1] if ipeak else None, vertical=False)
    sy = _strip(py, h, size, full, ipeak[0] if ipeak else None, vertical=True)
    out = np.full((h + size, w + size, 3), 18, np.uint8)
    out[:h, :w], out[:h, w:], out[h:, :w] = img, sy, sx
    out[h, :], out[:, w] = 60, 60
    return out


def seismograph(t: np.ndarray, I: np.ndarray, s: np.ndarray, now: float, minutes: float, width: int, height: int,
                marks: List[Tuple[float, str]], y_range: Sequence[float] = ()) -> np.ndarray:
    """Rolling I_tot(t) of the last `minutes`.

    Thousands of samples are compressed into `width` columns; a plain decimation would drop events lasting a few
    frames (a short beam trip would vanish), so each column draws the MIN and the MAX of its samples. Stretches with
    saturated pixels, where I_tot is under-estimated, are flagged with a red bar under the trace.
    """
    T = minutes * 60.0
    panel = np.full((height, width, 3), 18, np.uint8)
    top, bot = 22, 18
    ph = height - top - bot
    t0 = now - T
    for step in (5, 10, 15, 30, 60, 120, 300, 600, 900):
        if T / step <= 8:
            break
    k = 0
    while k * step <= T + 1e-9:
        x = int(round((T - k * step) / T * (width - 1)))
        panel[top:top + ph, x] = 45
        lab = "now" if k == 0 else f"-{int(k * step) // 60}:{int(k * step) % 60:02d}"
        cv2.putText(panel, lab, (max(0, min(width - 36, x - 14)), height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (150, 150, 150), 1, cv2.LINE_AA)
        k += 1
    keep = t >= t0
    t, I, s = t[keep], I[keep], s[keep]
    if t.size < 2:
        cv2.putText(panel, "I_tot: waiting for samples", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                    cv2.LINE_AA)
        return panel
    col = np.clip(((t - t0) / T * (width - 1)).astype(np.int64), 0, width - 1)
    starts = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
    cols = col[starts]
    lo, hi, sat = np.minimum.reduceat(I, starts), np.maximum.reduceat(I, starts), np.maximum.reduceat(s, starts)
    if len(y_range) == 2:
        vmin, vmax = float(y_range[0]), float(y_range[1])
    else:
        vmin, vmax = float(lo.min()), float(hi.max())
        pad = 0.05 * (vmax - vmin) if vmax > vmin else max(abs(vmax), 1.0) * 0.05
        vmin, vmax = vmin - pad, vmax + pad
    span = vmax - vmin if vmax > vmin else 1.0

    def ypix(v):
        return (top + ph - 1 - np.clip((v - vmin) / span, 0, 1) * (ph - 1)).astype(np.int32)

    cv2.polylines(panel, [np.stack([cols, ypix(hi)], 1).astype(np.int32), np.stack([cols, ypix(lo)], 1).astype(np.int32)],
                  False, (80, 220, 80), 1)
    if np.any(sat > 0):
        panel[top + ph:top + ph + 4, cols[sat > 0]] = (40, 40, 230)
    for tm, lab in marks:
        if tm >= t0:
            x = int(round((tm - t0) / T * (width - 1)))
            colr = (60, 60, 255) if lab == "REC" else (160, 160, 160)
            cv2.line(panel, (x, top), (x, top + ph), colr, 1)
            cv2.putText(panel, lab, (min(width - 40, x + 3), top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, colr, 1,
                        cv2.LINE_AA)
    txt = f"I_tot {I[-1]:.4g}   [{vmin:.3g} .. {vmax:.3g}]   {minutes:g} min"
    if s[-1] > 0:
        txt += f"   SAT {int(s[-1])} px"
    cv2.putText(panel, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if s[-1] > 0 else (210, 210, 210), 1,
                cv2.LINE_AA)
    return panel
