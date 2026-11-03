"""Quick beam readout for alignment: peak, saturated pixels, centre and FWHM.

For ALIGNMENT ONLY. Background is a percentile of the frame and the widths are FWHM read off the row and column
through the peak of a 3x3-smoothed image, not fitted sizes. With a saturated core the half level is computed from the
clip level instead of the true peak, so the FWHM is OVER-estimated; keep `saturated px` at 0 when the number matters.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def fwhm_1d(prof: np.ndarray, i_peak: int, half: float) -> float:
    """Width at level `half`, walking out from i_peak to the first crossing on each side (linear interpolation).
    Local by construction: a bright edge elsewhere in the frame cannot inflate it as a second moment would."""
    n = len(prof)
    if prof[i_peak] <= half:
        return float("nan")

    def cross(step: int) -> float:
        i = i_peak
        while 0 <= i + step < n and prof[i + step] > half:
            i += step
        j = i + step
        if not 0 <= j < n:
            return float(i)
        y1, y2 = prof[i], prof[j]
        return float(i) if y2 == y1 else i + step * (y1 - half) / (y1 - y2)

    return abs(cross(1) - cross(-1))


def beam_readout(ch: np.ndarray, mm_per_px: float, sat_level: float, bg_percentile: float = 20.0) -> Dict[str, Any]:
    """Peak, saturated pixels, centre (mm from image centre) and FWHM (mm) of the channel image `ch`."""
    a = ch.astype(np.float32)
    out: Dict[str, Any] = {"peak": float(a.max()), "n_sat": int((a >= sat_level).sum()) if sat_level else 0}
    sm = cv2.blur(a, (3, 3))                     # removes single hot pixels
    sm -= float(np.percentile(sm, bg_percentile))
    amp = float(sm.max())
    h, w = sm.shape
    if amp <= 0:
        out.update(xbar=float("nan"), ybar=float("nan"), fwhm_x=float("nan"), fwhm_y=float("nan"), ipeak=None)
        return out
    iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
    # On a saturated flat top argmax is an arbitrary plateau pixel: use the centroid of the plateau, searched only
    # near argmax so an unrelated bright object cannot drag the centre away.
    r = max(5, min(h, w) // 8)
    y0, y1, x0, x1 = max(0, iy - r), min(h, iy + r + 1), max(0, ix - r), min(w, ix + r + 1)
    top = sm[y0:y1, x0:x1] >= 0.98 * amp
    if top.sum() > 1:
        ys, xs = np.nonzero(top)
        iy, ix = int(round(ys.mean())) + y0, int(round(xs.mean())) + x0
    half = 0.5 * amp
    out.update(xbar=(ix - w / 2.0) * mm_per_px, ybar=(iy - h / 2.0) * mm_per_px,
               fwhm_x=fwhm_1d(sm[iy, :], ix, half) * mm_per_px, fwhm_y=fwhm_1d(sm[:, ix], iy, half) * mm_per_px,
               ipeak=(int(iy), int(ix)))
    return out


def profiles(ch: np.ndarray, mode: str = "peak", ipeak: Optional[Tuple[int, int]] = None):
    """(prof_x, prof_y), raw channel values (no background subtraction, so clipping shows as a flat top).
    peak: row and column through the peak (the cut the FWHM uses) | sum: mean projection along each axis."""
    a = ch.astype(np.float32)
    if mode == "sum":
        return a.mean(axis=0), a.mean(axis=1)
    iy, ix = ipeak if ipeak is not None else np.unravel_index(int(np.argmax(a)), a.shape)
    iy, ix = int(np.clip(iy, 0, a.shape[0] - 1)), int(np.clip(ix, 0, a.shape[1] - 1))
    return a[iy, :], a[:, ix]


class ValuesAverager:
    """Mean and sample std (ddof = 1) of N readouts. Averages already noise-biased widths: use frame averaging
    (in the capture engine) to reduce that bias; use this one to see the scatter of independent estimates."""

    KEYS = ("peak", "n_sat", "xbar", "ybar", "fwhm_x", "fwhm_y")

    def __init__(self, n: int):
        self.n, self.samples = max(1, int(n)), []

    def push(self, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.samples.append(r)
        if len(self.samples) < self.n:
            return None
        out = dict(self.samples[-1])
        for k in self.KEYS:
            v = np.array([s.get(k, np.nan) for s in self.samples], float)
            v = v[np.isfinite(v)]
            if v.size:
                out[k] = float(v.mean())
                out[k + "_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        out["n_avg"] = len(self.samples)
        self.samples = []
        return out
