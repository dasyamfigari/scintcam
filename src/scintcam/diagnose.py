"""scintcam diagnose: test what each camera control really does, with the stream running, and save a profile.

Reading a control back proves nothing: the uvcvideo driver caches the number written, and camera firmware may keep
reporting a stored number while the sensor uses another. Every test here judges a control by the IMAGE (mean level
and colour of the frames) and by the delivered frame rate:

    level   = mean luma of a frame [counts, 0-255]  (luma = 0.114 B + 0.587 G + 0.299 R)
    colour  = mean B, G, R of a frame [counts]
    noise   = standard deviation of the frame level on a static scene [counts]
    effect  = the change is larger than max(2 counts, 4 noise)

The scene must be lit, static and unsaturated (room lights on, nothing moving, the plate in view).
Results go to a JSON profile that scintcam reads at start (camera.profile) and to a text report next to it.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import __version__
from . import camera as C

log = logging.getLogger("scintcam")

SCALE = 1.0               # multiplies every wait (tests run a fast simulated camera with SCALE < 1)
PROFILE_VERSION = 1
UNIT_MS = 0.1             # UVC standard unit of exposure_absolute (100 us); many cameras differ: measured below
RATE_DROP = 0.95          # delivered rate below 95 % of the short-exposure rate counts as slowed down


def _sleep(s: float) -> None:
    time.sleep(s * SCALE)


# --------------------------------------------------------------------------------------------------------------
# Image meter
# --------------------------------------------------------------------------------------------------------------
class Meter:
    """Reads frames continuously from the open capture and turns them into level / colour / rate numbers."""

    def __init__(self, cap):
        self.cap = cap

    def _frames(self, seconds: float, n_min: int) -> List[Tuple[float, np.ndarray, float]]:
        out, t_end = [], time.monotonic() + seconds * SCALE
        while time.monotonic() < t_end or len(out) < n_min:
            ok, f = self.cap.read()
            t = time.monotonic()
            if time.monotonic() > t_end + 10 * SCALE:
                if not out:
                    raise IOError("the camera stopped delivering frames")
                break
            if not ok or f is None:
                continue
            small = f[::4, ::4].astype(np.float32)
            bgr = small.reshape(-1, 3).mean(0)
            sat = float((small.max(axis=2) >= 250).mean())
            out.append((t, bgr, sat))
            if len(out) > 400:
                break
        return out

    def drain(self, seconds: float, n_min: int = 3) -> None:
        """Keep reading (and discard) so that frames queued before a change are gone (>= n_min frames)."""
        self._frames(seconds, n_min)

    def measure(self, seconds: float = 0.6, n_min: int = 6) -> Dict[str, Any]:
        """level/colour/saturation means over the frames; fps = 1 / median frame interval (robust to read jitter)."""
        fr = self._frames(seconds, n_min)
        t = np.array([x[0] for x in fr])
        bgr = np.array([x[1] for x in fr])
        lv = bgr @ np.array([0.114, 0.587, 0.299])
        return {"level": float(lv.mean()), "noise": float(lv.std()), "bgr": [float(v) for v in bgr.mean(0)],
                "sat": float(np.mean([x[2] for x in fr])),
                "fps": float(1.0 / np.median(np.diff(t))) if len(t) > 2 and np.median(np.diff(t)) > 0 else 0.0,
                "frames": len(fr)}

    def trace(self, seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        """Per-frame (time since call, level) for a latency measurement."""
        t0 = time.monotonic()
        fr = self._frames(seconds, 3)
        return (np.array([x[0] - t0 for x in fr]) / SCALE,
                np.array([x[1] @ np.array([0.114, 0.587, 0.299]) for x in fr]))


# --------------------------------------------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------------------------------------------
class Diagnosis:
    def __init__(self, cfg: Dict[str, Any], cap, nominal_fps: float):
        self.cfg, self.cam, self.dev = cfg, cfg["camera"], cfg["camera"]["device"]
        self.m = Meter(cap)
        self.cap = cap
        self.fps = nominal_fps
        self.settle = 1.0                     # wait after a change [s]; replaced by the measured latency
        self.thr = 2.0                        # effect threshold [counts]; replaced after the noise measurement
        self.ctr = C.v4l2_controls(self.dev) or {}
        self.orig = json.loads(json.dumps(self.ctr))
        self.exp = next((n for n in C.CONTROL_MAP["exposure"][0] if n in self.ctr), None)
        self.ae = next((n for n in C.CONTROL_MAP["auto_exposure"][0] if n in self.ctr), None)
        self.findings: List[str] = []
        self.p: Dict[str, Any] = {}

    # ---------------------------------------------------------------- small helpers
    def say(self, msg: str) -> None:
        print("  " + msg, flush=True)

    def note(self, msg: str) -> None:
        self.findings.append(msg)
        print("  -> " + msg, flush=True)

    def set(self, name: str, value) -> Tuple[bool, str]:
        return C.v4l2_set_msg(self.dev, name, value)

    def get(self, name: str) -> Optional[int]:
        return C.v4l2_get(self.dev, name)

    def mode(self, v: int) -> None:
        if self.ae:
            self.set(self.ae, v)

    def look(self, settle: Optional[float] = None, seconds: float = 0.6) -> Dict[str, Any]:
        self.m.drain(self.settle if settle is None else settle)
        return self.m.measure(seconds)

    def robust(self, e: int) -> None:
        C.apply_exposure(self.dev, int(e), "full_sequence", self.cam["exposure_cycle_delay_s"] * SCALE)

    def clampe(self, e: float) -> int:
        d = self.ctr.get(self.exp, {})
        lo, hi = d.get("min", 1), d.get("max", 10000)
        return int(min(max(round(e), lo), hi))

    def effect(self, a: float, b: float) -> bool:
        return abs(a - b) > self.thr

    # ---------------------------------------------------------------- steps
    def identity(self) -> None:
        info = C.device_info(self.dev)
        fmt = C.v4l2_format(self.dev) or {}
        self.p["device"] = dict(info, path=C.v4l2_device(self.dev), kernel=platform.release(), format=fmt,
                                fps_nominal=round(self.fps, 3))
        self.p["raw"] = {"info": C.v4l2_query(self.dev, ["--info"]) or "",
                         "controls": C.v4l2_query(self.dev, ["-L"]) or C.v4l2_query(self.dev, ["-l"]) or "",
                         "formats": C.v4l2_query(self.dev, ["--list-formats-ext"]) or ""}
        self.say(f"device   {info.get('card', '?')}  ({info.get('driver', '?')} {info.get('driver_version', '')}, "
                 f"usb {info.get('usb_id', '?')}, kernel {platform.release()})")
        self.say(f"format   {fmt.get('pixel_format', '?')} {fmt.get('width', '?')}x{fmt.get('height', '?')}, "
                 f"{self.fps:.1f} fps when opened (before any setting)")

    def pin_automatics(self) -> None:
        """White balance and focus automatics off for the whole test (they would move the colour and the level)."""
        for key in ("auto_wb", "autofocus"):
            for n in C.CONTROL_MAP[key][0]:
                if n in self.ctr:
                    self.set(n, 0)
        self.ctr = C.v4l2_controls(self.dev) or self.ctr      # flags change (e.g. wb temperature now active)

    def working_point(self) -> Optional[int]:
        """Manual exposure giving a mid level (40-150 counts), so that halving or doubling it stays measurable."""
        d = self.ctr.get(self.exp, {})
        e = d.get("value") if isinstance(d.get("value"), int) else d.get("default", 100)
        e = self.clampe(e)                 # start from the camera's current value
        st = None
        for _ in range(10):
            self.robust(e)
            st = self.look()
            self.say(f"exposure {e:>7} -> level {st['level']:6.1f}")
            if st["level"] > 150 and e > d.get("min", 1):
                e = self.clampe(e / 3)
            elif st["level"] < 40 and e < d.get("max", 10000):
                e = self.clampe(e * 3)
            else:
                return e
        return e if st and 15 < st["level"] < 235 else None

    def noise(self) -> None:
        st = self.m.measure(1.5, 20)
        self.thr = max(2.0, 4 * st["noise"])
        self.p["scene"] = {"level": round(st["level"], 2), "noise": round(st["noise"], 3), "sat": round(st["sat"], 4),
                           "fps": round(st["fps"], 2), "effect_threshold": round(self.thr, 2)}
        self.say(f"scene    level {st['level']:.1f} counts, frame-to-frame noise {st['noise']:.2f}, "
                 f"saturated {100 * st['sat']:.2f} %, {st['fps']:.1f} fps measured")
        if st["sat"] > 0.01:
            self.note("more than 1 % of the scene is saturated: tests are less sensitive (dim the light or the scene)")

    def exposure_method(self, ea: int, eb: int) -> Dict[str, Any]:
        """Which writing method puts a new exposure into effect, judged by the image level."""
        r: Dict[str, Any] = {}
        self.robust(ea)
        la = self.look()["level"]
        self.robust(eb)
        lb = self.look()["level"]
        r["full_sequence"] = {"level_a": round(la, 2), "level_b": round(lb, 2)}
        self.say(f"full sequence: exposure {ea} -> level {la:.1f}, {eb} -> {lb:.1f}")
        if not self.effect(la, lb):
            r["apply"] = "none"
            self.note("the exposure control has NO visible effect, even with the full auto/manual sequence")
            return r

        def frac(level: float) -> float:
            return (level - la) / (lb - la)

        tests = (("direct", lambda: (self.mode(1), self.set(self.exp, eb))),
                 ("store_then_cycle", lambda: (self.mode(1), self.set(self.exp, eb), self.mode(3),
                                               _sleep(0.1), self.mode(1))),
                 ("write_in_auto", lambda: (self.mode(3), self.set(self.exp, eb), self.mode(1))))
        r["apply"] = "full_sequence"
        for name, act in tests:
            self.robust(ea)
            self.look()
            act()
            f = frac(self.look()["level"])
            rb = self.get(self.exp)
            r[name] = {"fraction": round(f, 3), "readback": rb}
            self.say(f"{name:<17} reached {100 * f:5.0f} % of the change (reads back {rb})")
            if f > 0.8 and r["apply"] == "full_sequence":
                r["apply"] = name
        if r["direct"]["fraction"] < 0.2 and r["direct"]["readback"] == eb:
            self.note("a manual exposure written in manual mode is STORED (it reads back) but NOT USED")
        self.note(f"exposure is put into effect by: {r['apply']}")
        return r

    def latency(self, ea: int, eb: int, method: str) -> float:
        C.apply_exposure(self.dev, ea, method, self.cam["exposure_cycle_delay_s"] * SCALE)
        la = self.look(1.0)["level"]
        t0 = time.monotonic()
        C.apply_exposure(self.dev, eb, method, self.cam["exposure_cycle_delay_s"] * SCALE)
        cmd = time.monotonic() - t0
        t, lv = self.m.trace(2.5)
        lb = lv[-5:].mean()
        ok = np.nonzero(np.abs(lv - la) >= 0.8 * abs(lb - la))[0]
        lat = cmd + (float(t[ok[0]]) if ok.size else 2.5)
        self.say(f"a new exposure shows in the image {lat:.2f} s after the command starts ({cmd:.2f} s of commands)")
        return lat

    def auto_behaviour(self, ea: int) -> Dict[str, Any]:
        """Auto mode: does the camera report the exposure it chose, and what does manual mode use afterwards."""
        r: Dict[str, Any] = {}
        g_manual = self.get("gain") if "gain" in self.ctr else None
        g_auto = None
        for stored in dict.fromkeys([ea, self.clampe(ea / 4), self.clampe(ea * 4)]):
            self.robust(stored)
            ls = self.look()["level"]
            self.mode(3)
            t, lv = self.m.trace(3.0)
            la = float(lv[-6:].mean())
            reads = []
            for _ in range(3):
                reads.append(self.get(self.exp))
                self.m.drain(0.3)
            if g_manual is not None:
                g_auto = self.get("gain")
            r.update(stored=stored, level_stored=round(ls, 2), level_auto=round(la, 2), readback_auto=reads)
            if self.effect(la, ls) and abs(la - ls) > 0.15 * ls:
                break
        settled = np.nonzero(np.abs(lv - la) <= max(self.thr, 0.1 * abs(la - ls)))[0]
        r["auto_settle_s"] = round(float(t[settled[0]]) if settled.size else 3.0, 2)
        distinct = self.effect(r["level_auto"], r["level_stored"])
        live = any(v is not None and v != r["stored"] for v in reads)
        r["auto_readback"] = "live" if live else ("stale" if distinct else "unknown")
        self.mode(1)
        lm = self.look()["level"]
        r["level_manual_after"] = round(lm, 2)
        r["readback_manual_after"] = self.get(self.exp)
        if not distinct:
            r["manual_after_auto"] = "unknown"
            self.note("auto exposure chose about the stored exposure: could not tell what manual mode keeps")
        else:
            keeps = abs(lm - r["level_auto"]) < abs(lm - r["level_stored"])
            r["manual_after_auto"] = "keeps_auto" if keeps else "returns_to_stored"
        self.say(f"auto: level {r['level_stored']:.1f} (stored exposure {r['stored']}) -> {r['level_auto']:.1f}"
                 f" in auto, reads {reads}; back in manual: level {lm:.1f}, reads {r['readback_manual_after']}")
        if r["auto_readback"] == "stale":
            self.note("in auto mode the camera reports the STORED manual exposure, not the one it uses: "
                      "the auto exposure cannot be read")
        elif r["auto_readback"] == "live":
            self.note("in auto mode the camera reports the exposure it is using")
        if r["manual_after_auto"] == "returns_to_stored":
            self.note("switching auto -> manual returns to the stored manual exposure (the auto value is lost)")
        elif r["manual_after_auto"] == "keeps_auto":
            self.note("switching auto -> manual keeps the exposure auto had chosen")
        if g_manual is not None:
            r["gain_manual"], r["gain_auto"] = g_manual, g_auto
            if g_auto is not None and g_auto != g_manual:
                self.note(f"auto exposure also changes the gain ({g_manual} -> {g_auto})")
        return r

    def _point(self, e: int, method: str) -> Dict[str, Any]:
        C.apply_exposure(self.dev, e, method, self.cam["exposure_cycle_delay_s"] * SCALE)
        st = self.look(self.settle, 0.8)
        self.say(f"exposure {e:>7}  level {st['level']:6.1f}  saturated {100 * st['sat']:5.1f} %  {st['fps']:5.1f} fps")
        return {"exposure": int(e), "level": round(st["level"], 2), "sat": round(st["sat"], 4),
                "fps": round(st["fps"], 3)}

    def scan(self, method: str, ea: int) -> Dict[str, Any]:
        """Level, saturation and delivered rate against exposure; exposure unit from the frame rate.

        Once the exposure is longer than the frame period the sensor must slow down, and then
            T_frame = 1 / fps = u E + d
        T_frame frame period [s], E exposure in camera units, u duration of one unit [s] (UVC standard: 100 us),
        d fixed readout/blanking time [s]. The slope u is the real exposure unit."""
        d = self.ctr.get(self.exp, {})
        top = d.get("max", 10000)
        lo = self.clampe(max(d.get("min", 1), ea / 16))
        hi = self.clampe(min(top, ea * 8))
        vals = sorted(set(self.clampe(v) for v in np.geomspace(max(lo, 1), max(hi, lo + 1), 10)))
        rows = [self._point(e, method) for e in vals]
        f0 = float(np.median([x["fps"] for x in rows[:3]]))
        for _ in range(6):           # extend until the rate drops (needed for the unit), frames shorter than ~0.5 s
            last = rows[-1]
            if last["fps"] < RATE_DROP * f0 or last["exposure"] >= top or last["fps"] < 2:
                break
            rows.append(self._point(self.clampe(last["exposure"] * 2), method))
        r: Dict[str, Any] = {"scan": rows, "fps_full": round(f0, 3)}
        k = len(rows)                 # slow region = the longest exposures, all below RATE_DROP * f0
        while k > 0 and rows[k - 1]["fps"] < RATE_DROP * f0:
            k -= 1
        fast, slow = rows[:k], rows[k:]
        a_ = max(x["exposure"] for x in fast) if fast else None
        b_ = slow[0]["exposure"] if slow else None
        if slow and fast:            # bisect between the last full-rate and the first slow exposure
            for _ in range(4):
                if b_ - a_ <= 1:
                    break
                mid = (a_ + b_) // 2
                C.apply_exposure(self.dev, mid, method, self.cam["exposure_cycle_delay_s"] * SCALE)
                f = self.look(self.settle, 0.8)["fps"]
                a_, b_ = (mid, b_) if f >= RATE_DROP * f0 else (a_, mid)
        pts = [(x["exposure"], 1.0 / x["fps"]) for x in slow if x["fps"] > 0]
        if b_:
            # points from just above the limit outwards: those near the limit pin the crossing with the full-rate
            # frame period, the far ones the slope (u and d are strongly anticorrelated if only far points are used)
            for fac in (1.08, 1.15, 1.25, 1.4, 1.7, 2.2, 3.0):
                e = self.clampe(b_ * fac)
                if e not in [p[0] for p in pts] and e > b_:
                    st = self._point(e, method)
                    if st["fps"] > 0 and st["fps"] < RATE_DROP * f0:
                        pts.append((e, 1.0 / st["fps"]))
        unit, src = UNIT_MS, "UVC standard (not measured: the frame rate never dropped)"
        if len(pts) >= 2 and max(p[0] for p in pts) > 1.15 * min(p[0] for p in pts):
            x = np.array([p[0] for p in pts], float)
            y = np.array([p[1] for p in pts]) / SCALE
            if len(pts) >= 4:
                (u, dd), cov = np.polyfit(x, y, 1, cov=True)
            else:
                (u, dd), cov = np.polyfit(x, y, 1), np.full((2, 2), np.nan)
            r2 = 1 - ((y - (u * x + dd)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-18)
            u, dd = float(u), float(dd)
            # limit E_lim = (T0 - d) / u, T0 = 1/f0; its variance from the fit covariance:
            #   var(E_lim) = [var(d) + E_lim^2 var(u) + 2 E_lim cov(u, d)] / u^2
            t0 = 1.0 / f0 / SCALE
            e_lim = (t0 - dd) / u
            var_lim = (cov[1, 1] + e_lim ** 2 * cov[0, 0] + 2 * e_lim * cov[0, 1]) / u ** 2
            r["unit_fit"] = {"points": len(pts), "unit_us": round(u * 1e6, 4),
                             "unit_us_err": round(float(np.sqrt(cov[0, 0])) * 1e6, 4) if np.isfinite(cov[0, 0]) else None,
                             "readout_ms": round(dd * 1e3, 3),
                             "readout_ms_err": round(float(np.sqrt(cov[1, 1])) * 1e3, 3) if np.isfinite(cov[1, 1]) else None,
                             "limit_ms_err": round(float(np.sqrt(var_lim)) * u * 1e3, 3) if np.isfinite(var_lim) else None,
                             "r2": round(float(r2), 5)}
            f = r["unit_fit"]
            self.say(f"frame period = ({u * 1e6:.4f} +- {f['unit_us_err']}) us x exposure + ({dd * 1e3:.2f} +- "
                     f"{f['readout_ms_err']}) ms   (R2 {r2:.4f}, {len(pts)} points; limit +- {f['limit_ms_err']} ms)")
            if u > 0 and r2 > 0.98:
                unit, src = u * 1e3, "measured from the frame rate"
                r["unit_ms_measured"] = round(unit, 7)
                # longest exposure that still fits the full-rate frame: E_lim = (T_frame - d) / u, with T_frame = 1/f0
                lim = int(e_lim)
                if a_ is not None and b_ is not None and 0.8 * a_ <= lim <= b_:
                    a_ = lim
        r["unit_ms"], r["unit_source"] = round(unit, 7), src
        for x in rows:
            x["ms"] = round(x["exposure"] * unit, 3)
        if "unit_ms_measured" in r:
            err = r["unit_fit"].get("unit_us_err")
            msg = (f"1 exposure unit = {unit * 1e3:.3f}{f' +- {err:.3f}' if err else ''} us "
                   f"(measured from the frame rate; UVC standard: 100 us)")
            if abs(unit / UNIT_MS - 1) > 0.1:
                msg += f": this camera does NOT use 100 us; 'exposure / 10 ms' is wrong by x{UNIT_MS / unit:.1f}"
            self.note(msg)
        else:
            self.note("exposure unit not measured (the frame rate never dropped): ms values assume 100 us units")
        if slow:
            r["full_rate_max"], r["rate_drops_from"] = a_, b_
            r["full_rate_max_ms"] = round(a_ * unit, 3) if a_ else None
            le = (r.get("unit_fit") or {}).get("limit_ms_err")
            self.note(f"exposure above {a_} ({a_ * unit:.1f}{f' +- {le:.1f}' if le else ''} ms) lowers the frame rate "
                      f"({f0:.1f} fps -> {slow[-1]['fps']:.1f} fps at {slow[-1]['exposure']})" if a_ else
                      "the frame rate drops at every exposure tested")
        sat = [x for x in rows if x["sat"] > 0.01 or x["level"] > 245]
        r["saturates_from"] = sat[0]["exposure"] if sat else None
        good = [x for x in rows if x["sat"] <= 0.005 and 5 < x["level"] < 240]
        if len(good) >= 3:
            e = np.array([x["ms"] for x in good])
            lv = np.array([x["level"] for x in good])
            b, a = np.polyfit(e, lv, 1)
            r2 = 1 - ((lv - (a + b * e)) ** 2).sum() / max(((lv - lv.mean()) ** 2).sum(), 1e-9)
            r["response"] = {"offset": round(float(a), 3), "slope_per_ms": round(float(b), 4), "r2": round(float(r2), 4)}
            pos = lv - a > 1
            if pos.sum() >= 3:
                r["response"]["power"] = round(float(np.polyfit(np.log(e[pos]), np.log(lv[pos] - a), 1)[0]), 3)
            self.say(f"level = {a:.2f} + {b:.4f} x exposure[ms]  (R2 {r2:.4f}, "
                     f"power {r['response'].get('power', float('nan')):.2f})")
            if r2 < 0.98:
                self.note("level is not proportional to exposure (gamma or another curve is on): "
                          "intensities are not linear in light")
        if len(set(x["level"] for x in rows)) == 1:
            self.note("the scan shows no change at all")
        return r

    def out_of_range(self, name: str) -> str:
        d = self.ctr.get(name, {})
        lo, hi, v0 = d.get("min"), d.get("max"), self.get(name)
        if not isinstance(hi, int) or not isinstance(lo, int):
            return "unknown"
        big = hi + max(1, (hi - lo) // 2)
        ok, msg = self.set(name, big)
        rb = self.get(name)
        if v0 is not None:
            self.set(name, v0)
        if not ok:
            return "error"
        return "clamped" if rb == hi else ("accepted" if rb == big else f"read back {rb}")

    def other_controls(self, ea: int, method: str) -> Dict[str, Any]:
        """Each integer control in turn: change it by a quarter of its range and watch level and colour."""
        out: Dict[str, Any] = {}
        skip = set(C.CONTROL_MAP["exposure"][0])
        if self.exp and ea is not None:
            C.apply_exposure(self.dev, ea, method, self.cam["exposure_cycle_delay_s"] * SCALE)
        base = self.look(1.0)
        for name, d in self.ctr.items():
            if name in skip or "min" not in d or "max" not in d or "value" not in d:
                continue
            flags = str(d.get("flags", ""))
            entry: Dict[str, Any] = {k: d.get(k) for k in ("min", "max", "step", "default", "value")}
            entry["flags"] = flags or None
            entry["type"] = d.get("type")
            if d.get("type") not in (None, "int") or "inactive" in flags or "read-only" in flags or d["max"] - d["min"] < 2 or d["max"] - d["min"] > 1e6:
                entry["tested"] = False
                out[name] = entry
                continue
            v0, span = d["value"], d["max"] - d["min"]
            v1 = v0 + span // 4 if v0 + span // 4 <= d["max"] else v0 - span // 4
            ok, msg = self.set(name, v1)
            st = self.look()
            rb = self.get(name)
            self.set(name, v0)
            back = self.look()
            dl = st["level"] - base["level"]
            dc = max(abs((st["bgr"][i] - base["bgr"][i]) - dl) for i in range(3))
            effect = "level" if abs(dl) > self.thr else ("colour" if dc > self.thr else "none")
            if not ok:
                effect = "write refused"
            entry.update(tested=True, test_value=v1, accepted=ok, message=msg or None, readback=rb,
                         readback_ok=(rb == v1), delta_level=round(dl, 2), delta_colour=round(dc, 2), effect=effect,
                         reversible=not self.effect(back["level"], base["level"]),
                         fps_change=round(st["fps"] - base["fps"], 2))
            entry["out_of_range"] = self.out_of_range(name)
            self.say(f"{name:<28} {v0} -> {v1}: level {dl:+6.1f}, colour {dc:5.1f}  => {effect}"
                     f"{'' if entry['reversible'] else ', NOT reversible'}; out of range: {entry['out_of_range']}")
            out[name] = entry
            base = back
        none = [n for n, e in out.items() if e.get("effect") == "none"]
        if none:
            self.note("no visible effect on this scene: " + ", ".join(none))
        acc = [n for n, e in out.items() if e.get("out_of_range") == "accepted"]
        if acc:
            self.note("values outside the declared range are accepted (not clamped) by: " + ", ".join(acc))
        return out

    def reopen(self, ea: int, eb: int, method: str) -> Dict[str, Any]:
        """Do the settings survive closing and reopening the stream? (some firmware forgets manual mode)"""
        C.apply_exposure(self.dev, eb, method, self.cam["exposure_cycle_delay_s"] * SCALE)
        before = self.look(1.0)["level"]
        rb_before = self.get(self.exp)
        self.m.cap.release()
        _sleep(0.5)
        cap, _size, _fps = C.open_camera(self.cam)
        self.m.cap = self.cap = cap
        after = self.look(1.5)["level"]
        r = {"level_before": round(before, 2), "level_after": round(after, 2), "readback_before": rb_before,
             "readback_after": self.get(self.exp), "mode_after": self.get(self.ae) if self.ae else None}
        r["kept"] = not self.effect(before, after) and r["mode_after"] in (1, None)
        self.say(f"reopen: level {before:.1f} -> {after:.1f}, exposure reads {rb_before} -> {r['readback_after']}, "
                 f"mode {r['mode_after']}")
        if not r["kept"]:
            self.note("reopening the stream changes the exposure state: settings must be applied after every start "
                      "(scintcam does this)")
        return r

    def restore(self) -> None:
        """Put back every writable control as it was before the test."""
        ae0 = (self.orig.get(self.ae) or {}).get("value") if self.ae else None
        self.mode(1)
        for name, d in self.orig.items():
            if name in (self.ae, self.exp) or "value" not in d or "read-only" in str(d.get("flags", "")):
                continue
            if self.get(name) != d["value"]:
                self.set(name, d["value"])
        e0 = (self.orig.get(self.exp) or {}).get("value") if self.exp else None
        if e0 is not None:
            self.robust(e0)
        if ae0 is not None and ae0 != 1:
            self.mode(ae0)

    def run(self) -> Dict[str, Any]:
        self.p.update(profile_version=PROFILE_VERSION, scintcam=__version__,
                      created=datetime.datetime.now().isoformat(timespec="seconds"))
        print("\n[1/7] device", flush=True)
        self.identity()
        if not self.exp or not self.ae:
            self.note("this camera has no V4L2 exposure / auto-exposure controls: only the other controls are tested")
        self.pin_automatics()
        ex: Dict[str, Any] = {"control": self.exp, "auto_control": self.ae, "unit_ms": UNIT_MS,
                              "unit_source": "UVC standard (not measured)",
                              "declared_min": self.ctr.get(self.exp, {}).get("min"),
                              "declared_max": self.ctr.get(self.exp, {}).get("max")}
        self.p["exposure"] = ex
        method, ea = "full_sequence", None
        if self.exp and self.ae:
            print("\n[2/7] finding a usable exposure (full auto/manual sequence)", flush=True)
            ea = self.working_point()
            if ea is None:
                self.note("no exposure gave a usable level (15-235 counts): check the light and that the lens is open")
        print("\n[3/7] scene noise", flush=True)
        self.noise()
        if ea is not None:
            eb = self.clampe(ea / 2) if self.clampe(ea / 2) != ea else self.clampe(ea * 2)
            print("\n[4/7] how a manual exposure is put into effect", flush=True)
            ex["tests"] = self.exposure_method(ea, eb)
            method = ex["tests"]["apply"]
            ex["apply"] = method
            if method != "none":
                use = method
                ex["settle_s"] = round(min(2.0, max(0.3, self.latency(ea, eb, use) + 0.2)), 2)
                self.settle = ex["settle_s"]
                print("\n[5/7] auto exposure", flush=True)
                ex.update(self.auto_behaviour(ea))
                print("\n[6/7] exposure scan", flush=True)
                ex.update(self.scan(use, ea))
                ex["out_of_range"] = self.out_of_range(self.exp)
                self.say(f"exposure beyond the declared maximum: {ex['out_of_range']}")
                if ex["out_of_range"] == "accepted":
                    self.note("exposure values above the declared maximum are accepted and read back: "
                              "the read-back number is not the exposure in use")
                ex["reopen"] = self.reopen(ea, eb, use)
            ex["working_point"] = ea
        else:
            ex["apply"] = "none" if self.exp else None
        print("\n[7/7] other controls", flush=True)
        self.p["controls"] = self.other_controls(ea, method if method != "none" else "full_sequence")
        self.p["findings"] = self.findings
        return self.p


def report_text(p: Dict[str, Any]) -> str:
    """Human-readable summary of a profile."""
    d, ex = p.get("device", {}), p.get("exposure", {})
    L = [f"scintcam camera diagnosis  {p.get('created', '')}  (scintcam {p.get('scintcam', '')})",
         f"device  {d.get('card', '?')}  driver {d.get('driver', '?')} {d.get('driver_version', '')}  "
         f"usb {d.get('usb_id', '?')}  bus {d.get('bus_info', '?')}  kernel {d.get('kernel', '?')}",
         f"format  {d.get('format', {}).get('pixel_format', '?')} {d.get('format', {}).get('width', '?')}x"
         f"{d.get('format', {}).get('height', '?')}  {d.get('fps_nominal', '?')} fps when opened, "
         f"{(p.get('exposure') or {}).get('fps_full', '?')} fps at short exposure", ""]
    sc = p.get("scene") or {}
    L.append(f"scene   level {sc.get('level')}  noise {sc.get('noise')}  effect threshold {sc.get('effect_threshold')} counts")
    L += ["", "FINDINGS"] + [f"  - {f}" for f in p.get("findings", [])] + ["", "EXPOSURE"]
    for k in ("control", "auto_control", "unit_ms", "unit_source", "unit_fit", "declared_min", "declared_max",
              "apply", "settle_s", "auto_readback", "manual_after_auto", "auto_settle_s", "fps_full", "full_rate_max",
              "full_rate_max_ms", "rate_drops_from", "saturates_from", "out_of_range", "response"):
        if k in ex:
            L.append(f"  {k:<20} {ex[k]}")
    if ex.get("scan"):
        L.append("  scan: exposure        ms    level   sat %    fps")
        L += [f"        {r['exposure']:>8} {r['ms']:9.3f} {r['level']:8.1f} {100 * r['sat']:7.2f} {r['fps']:6.2f}"
              for r in ex["scan"]]
    L += ["", "OTHER CONTROLS  (change by 1/4 of the range; effect on this scene)",
          f"  {'control':<28} {'min':>6} {'max':>6} {'value':>6}  {'effect':<14} readback  out-of-range"]
    for n, e in (p.get("controls") or {}).items():
        if e.get("tested"):
            L.append(f"  {n:<28} {e['min']!s:>6} {e['max']!s:>6} {e['value']!s:>6}  {e['effect']:<14} "
                     f"{'ok' if e['readback_ok'] else 'differs':<9} {e.get('out_of_range')}")
        else:
            why = e.get("type") if e.get("type") not in (None, "int") else (e.get("flags") or "range")
            L.append(f"  {n:<28} {e.get('min')!s:>6} {e.get('max')!s:>6} {e.get('value')!s:>6}  (not tested: {why})")
    L += ["", "RAW v4l2-ctl --info", p.get("raw", {}).get("info", ""), "RAW v4l2-ctl -L",
          p.get("raw", {}).get("controls", "")]
    return "\n".join(L)


def run(cfg: Dict[str, Any], out_path: str = "") -> int:
    if not C.have_v4l2ctl():
        print("scintcam diagnose needs Linux and v4l2-ctl (package v4l-utils)")
        return 2
    out_path = out_path or cfg["camera"].get("profile") or "camera_profile.json"
    print("scintcam diagnose: about 3 minutes. Room lights ON, scene static (plate in view, nothing moving),\n"
          "scintcam and any other camera program closed. The camera settings are restored at the end.", flush=True)
    cap, _size, fps = C.open_camera(cfg["camera"])
    dg = Diagnosis(cfg, cap, fps)
    prof = None
    try:
        prof = dg.run()
    except KeyboardInterrupt:
        print("\ninterrupted: no profile written")
    finally:
        print("\nrestoring the camera settings ...", flush=True)
        dg.restore()
        dg.m.cap.release()
    if prof is None:
        return 1
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=1)
    os.replace(tmp, out_path)
    txt = os.path.splitext(out_path)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(report_text(prof) + "\n")
    print(f"\nprofile written: {out_path}  (report: {txt})")
    print("FINDINGS:\n" + "\n".join(f"  - {x}" for x in prof.get("findings", [])))
    return 0
