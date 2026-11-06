"""Camera access: open with a fallback mode ladder, pin automatic controls, read back the real driver state.

A webcam's firmware is built to produce a pleasant picture, not a measurement. For profiling a scintillator screen
the harmful automatics are, in order: auto-exposure/auto-gain (it reacts to the beam turning on, so the recorded
intensity follows the control loop instead of the screen), auto white balance (re-weights the colour channels frame
by frame), autofocus (changes the magnification, hence mm/px), sharpness (an edge filter that biases widths) and
backlight compensation (a hidden, spatially varying gain). None of them can be undone afterwards, so they are pinned
here and the state actually in use is read back from the driver and logged with every recording.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import controls as cfg_controls

log = logging.getLogger("scintcam")
IS_LINUX = sys.platform.startswith("linux")

BACKENDS = {"": cv2.CAP_ANY, "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
            "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)}

# config key -> (V4L2 names, legacy first then current kernels; OpenCV property or None)
CONTROL_MAP = {
    "auto_exposure": (["exposure_auto", "auto_exposure"], cv2.CAP_PROP_AUTO_EXPOSURE),
    "exposure": (["exposure_absolute", "exposure_time_absolute"], cv2.CAP_PROP_EXPOSURE),
    "gain": (["gain"], cv2.CAP_PROP_GAIN),
    "auto_wb": (["white_balance_temperature_auto", "white_balance_automatic"], cv2.CAP_PROP_AUTO_WB),
    "wb_temperature": (["white_balance_temperature"], cv2.CAP_PROP_WB_TEMPERATURE),
    "autofocus": (["focus_auto", "focus_automatic_continuous"], cv2.CAP_PROP_AUTOFOCUS),
    "focus": (["focus_absolute"], cv2.CAP_PROP_FOCUS),
    "brightness": (["brightness"], cv2.CAP_PROP_BRIGHTNESS),
    "contrast": (["contrast"], cv2.CAP_PROP_CONTRAST),
    "saturation": (["saturation"], cv2.CAP_PROP_SATURATION),
    "gamma": (["gamma"], cv2.CAP_PROP_GAMMA),
    "sharpness": (["sharpness"], cv2.CAP_PROP_SHARPNESS),
    "backlight": (["backlight_compensation"], cv2.CAP_PROP_BACKLIGHT),
    "power_line_frequency": (["power_line_frequency"], None),
}
SWITCHES = ("auto_exposure", "auto_wb", "autofocus")   # written first: manual values are rejected while auto is on


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


# --------------------------------------------------------------------------------------------------------------
# v4l2-ctl helpers (Linux)
# --------------------------------------------------------------------------------------------------------------
def have_v4l2ctl() -> bool:
    return IS_LINUX and shutil.which("v4l2-ctl") is not None


def v4l2_device(dev) -> str:
    return dev if isinstance(dev, str) else f"/dev/video{int(dev)}"


def v4l2_query(dev, args: List[str]) -> Optional[str]:
    try:
        p = subprocess.run(["v4l2-ctl", "-d", v4l2_device(dev)] + args, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return p.stdout if p.returncode == 0 else None


def v4l2_set_msg(dev, name: str, value) -> Tuple[bool, str]:
    """Write one control; (accepted, driver message). Accepted only means the driver took the number."""
    try:
        p = subprocess.run(["v4l2-ctl", "-d", v4l2_device(dev), "-c", f"{name}={value}"],
                           capture_output=True, text=True, timeout=5)
    except Exception as e:
        return False, str(e)
    msg = (p.stderr or "").strip()
    return p.returncode == 0 and not msg, msg


def v4l2_set(dev, name: str, value) -> bool:
    return v4l2_set_msg(dev, name, value)[0]


def v4l2_get(dev, name: str) -> Optional[int]:
    """Value the driver reports for one control (v4l2-ctl -C); None if it cannot be read."""
    txt = v4l2_query(dev, ["-C", name])
    for line in (txt or "").splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() == name:
            try:
                return int(v.strip())
            except ValueError:
                return None
    return None


def v4l2_controls(dev) -> Optional[Dict[str, Dict[str, Any]]]:
    """Every control as {name: {value, min, max, default, flags, ...}}, parsed from v4l2-ctl -l."""
    txt = v4l2_query(dev, ["-l"])
    if not txt:
        return None
    out = {}
    for line in txt.splitlines():
        head, sep, rest = line.partition(":")
        parts = head.split()
        if not sep or len(parts) < 2 or not parts[1].startswith("0x"):   # section titles, menu items
            continue
        d = {"type": parts[2].strip("()")} if len(parts) > 2 and parts[2].startswith("(") else {}
        for kv in rest.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    v = int(v)
                except ValueError:
                    pass
                d[k] = v
        out[parts[0]] = d
    return out


def v4l2_driver_fps(dev) -> Optional[float]:
    txt = v4l2_query(dev, ["--get-parm"])
    for line in (txt or "").splitlines():
        if "Frames per second" in line:
            try:
                return float(line.split(":")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def v4l2_format(dev) -> Optional[Dict[str, Any]]:
    txt = v4l2_query(dev, ["--get-fmt-video"])
    if not txt:
        return None
    out: Dict[str, Any] = {}
    for line in txt.splitlines():
        k, sep, v = (x.strip() for x in line.partition(":"))
        if not sep:
            continue
        if k == "Width/Height" and "/" in v:
            try:
                out["width"], out["height"] = (int(x) for x in v.split("/")[:2])
            except ValueError:
                pass
        elif k == "Pixel Format":
            out["pixel_format"] = v.split()[0].strip("'")
    return out


def device_info(dev) -> Dict[str, str]:
    """Identity of the device from v4l2-ctl --info: driver, card, bus_info, driver_version (+ USB id from sysfs)."""
    out: Dict[str, str] = {}
    keys = {"Driver name": "driver", "Card type": "card", "Bus info": "bus_info", "Driver version": "driver_version"}
    for line in (v4l2_query(dev, ["--info"]) or "").splitlines():
        k, sep, v = (x.strip() for x in line.partition(":"))
        if sep and k in keys and keys[k] not in out:
            out[keys[k]] = v
    node = os.path.basename(v4l2_device(dev))
    for up in ("../idVendor", "../../idVendor"):          # /sys/class/video4linux/videoN/device -> USB interface
        f = os.path.join("/sys/class/video4linux", node, "device", up)
        try:
            vid = open(f).read().strip()
            pid = open(f.replace("idVendor", "idProduct")).read().strip()
            out["usb_id"] = f"{vid}:{pid}"
            break
        except OSError:
            continue
    return out


def load_profile(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the camera profile written by `scintcam diagnose` into cfg["_profile"] (None if absent or for another
    camera). Without a profile scintcam uses the methods that work on most cameras."""
    cfg["_profile"] = None
    path = cfg["camera"].get("profile") or ""
    if not path:
        return None
    if not os.path.isabs(path):      # next to the config file first, then the folder scintcam was started from
        cands = ([os.path.join(os.path.dirname(cfg["_path"]), path)] if cfg.get("_path") else []) + [path]
        path = next((c for c in cands if os.path.exists(c)), path)
    if not os.path.exists(path):
        log.warning("no camera profile (%s, looked next to the config and in %s): exposures are shown in camera "
                    "steps; run `scintcam diagnose` once to measure the unit", path, os.getcwd())
        return None
    try:
        with open(path, encoding="utf-8") as f:
            prof = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("camera profile %s unreadable (%s): ignored", path, e)
        return None
    card = (prof.get("device") or {}).get("card")
    cur = device_info(cfg["camera"]["device"]).get("card") if have_v4l2ctl() else None
    if card and cur and card != cur:
        log.warning("camera profile %s was made for '%s' but this device is '%s': ignored (run scintcam diagnose)",
                    path, card, cur)
        return None
    ex = prof.get("exposure") or {}
    log.info("camera profile %s (%s, %s): exposure applied by '%s', auto value %s, manual after auto %s",
             path, card or "?", (prof.get("created") or "")[:10], ex.get("apply", "?"),
             ex.get("auto_readback", "?"), ex.get("manual_after_auto", "?"))
    cfg["_profile"] = prof
    return prof


# --------------------------------------------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------------------------------------------
def mode_ladder(cam: Dict[str, Any]) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Candidate (fourcc, width, height) modes, requested first, then standard webcam modes.

    A non-standard geometry (e.g. 480x480) is the usual reason a camera opens but never streams: the driver accepts
    the request, the buffers never fill, and every read ends in a 10 s V4L2 select() timeout.
    """
    w, h, fcc = int(cam["width"]), int(cam["height"]), cam["fourcc"] or ""
    ladder = [(fcc, w, h), ("" if fcc else "MJPG", w, h), ("MJPG", 640, 480), ("", 640, 480),
              ("MJPG", 1280, 720), ("", 1280, 720), ("", 320, 240), ("", None, None)]
    seen, out = set(), []
    for m in ladder:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _try_mode(cap, fourcc: str, w, h, fps: float, timeout: float):
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, _fourcc(fourcc))
    if w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame          # only a real frame proves the mode streams
    return None


def _video_capture(cam: Dict[str, Any]):
    dev = cam["device"]
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)
    return cv2.VideoCapture(dev, BACKENDS.get(cam["backend"], cv2.CAP_ANY))


def measure_fps(cap, n: int = 20) -> float:
    """Delivered frame rate over n reads: (n - 1) / (t_last - t_first)."""
    t = []
    for _ in range(n):
        ok, _f = cap.read()
        if ok:
            t.append(time.monotonic())
    return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0


def open_camera(cam: Dict[str, Any]):
    """Open the camera and return (cap, (width, height), nominal_fps). Raises IOError if no mode streams."""
    log.info("opening camera %s ...", cam["device"])
    cap = _video_capture(cam)
    if not cap.isOpened():
        raise IOError(f"could not open camera {cam['device']!r}: check the device, permissions (group 'video') "
                      "and that no other program holds it")
    frame = None
    for i, (fcc, w, h) in enumerate(mode_ladder(cam)):
        if i > 0:
            if not cam["fallback"]:
                break
            log.warning("no frames in the requested mode; trying %s %s", fcc or "driver format",
                        f"{w}x{h}" if w else "default size")
        frame = _try_mode(cap, fcc, w, h, cam["fps"], cam["open_timeout_s"])
        if frame is not None:
            break
    if frame is None:
        cap.release()
        raise IOError(f"camera {cam['device']!r} opened but never delivered a frame; run `scintcam probe` to see "
                      "which modes it really streams")
    h, w = frame.shape[:2]
    if (w, h) != (int(cam["width"]), int(cam["height"])):
        log.warning("requested %dx%d, the camera delivers %dx%d: update camera.width/height",
                    cam["width"], cam["height"], w, h)
    if cam["fps"]:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not 0 < fps <= 240:
            fps = float(cam["fps"])
    else:   # fps = 0: nothing was requested; the rate is whatever the camera delivers (e.g. set by its exposure)
        fps = measure_fps(cap) or cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info("camera streaming %dx%d, %.1f fps%s", w, h, fps, "" if cam["fps"] else " (first measurement, before the controls are set)")
    return cap, (w, h), float(fps)


def probe(cam: Dict[str, Any]) -> str:
    """Report the modes the driver lists and which of them actually deliver frames."""
    out = [f"camera probe: device {cam['device']}"]
    if have_v4l2ctl():
        out.append(v4l2_query(cam["device"], ["--list-formats-ext"]) or "(v4l2-ctl --list-formats-ext failed)")
    cap = _video_capture(cam)
    if not cap.isOpened():
        return "\n".join(out + ["could not open the device"])
    out.append(f"{'format':<8} {'requested':<12} {'obtained':<12} result")
    for fcc, w, h in mode_ladder(cam):
        f = _try_mode(cap, fcc, w, h, cam["fps"], 3.0)
        got = f"{f.shape[1]}x{f.shape[0]}" if f is not None else "-"
        out.append(f"{fcc or '-':<8} {f'{w}x{h}' if w else 'default':<12} {got:<12} {'ok' if f is not None else 'NO FRAMES'}")
    cap.release()
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------------------
# Locking
# --------------------------------------------------------------------------------------------------------------
def _v4l2_value(key: str, val):
    if key == "auto_exposure":
        return 1 if val is False else 3          # V4L2 exposure_auto: 1 = manual, 3 = aperture priority (auto)
    if isinstance(val, bool):
        return int(val)
    return val


def lock_camera(cap, cfg: Dict[str, Any]) -> None:
    """Pin the controls listed in [camera.controls]. Must be called with frames flowing (UVC rejects some otherwise)."""
    cam = cfg["camera"]
    if not cam["lock"]:
        log.warning("camera.lock = false: automatic controls left on; intensities are not comparable between frames")
        return
    ctl = cfg_controls(cfg)
    use_v4l2 = have_v4l2ctl() if cam["use_v4l2ctl"] == "auto" else bool(cam["use_v4l2ctl"])
    done = set()
    if use_v4l2:
        done = _lock_v4l2(cam, ctl, exposure_strategy(cfg), exposure_unit_ms(cfg))
    elif IS_LINUX:
        log.warning("v4l2-ctl not found (install v4l-utils): controls set through OpenCV only, less reliable")
    _lock_opencv(cap, ctl, skip=done)


EXPOSURE_STRATEGIES = ("direct", "store_then_cycle", "write_in_auto", "full_sequence")


def apply_exposure(dev, value: int, strategy: str = "full_sequence", delay_s: float = 0.35) -> None:
    """Put a manual exposure into effect (value in camera steps; duration of one step from the profile, see
    exposure_unit_ms), by the method the camera needs.

    direct            write the value in manual mode (cameras that apply manual writes at once)
    store_then_cycle  write it (stored, not used), then a short auto -> manual transition, which latches the STORED
                      value (cameras that apply a manual exposure only on that transition)
    write_in_auto     switch to auto, write the value there, switch to manual
    full_sequence     auto, wait, manual, write, short auto -> manual, write again: works on all three kinds above,
                      used when no camera profile says which kind this camera is
    `scintcam diagnose` finds the method by watching the image; reading the value back proves nothing.

    Every write is preceded by a write of a neighbouring value (value - 1, or value + 1 at 1): a camera that switched
    auto -> manual keeping the auto exposure may still hold the target as its STORED number, and a write equal to
    the stored number is not guaranteed to reach the sensor. Changing the number first makes the final write a real
    change. (At the declared minimum the driver clamps value - 1 back to the same number.)
    """
    nudge = int(value) - 1 if int(value) > 1 else int(value) + 1
    auto = lambda v: any(v4l2_set(dev, n, v) for n in CONTROL_MAP["auto_exposure"][0])

    def expo() -> bool:
        for n in CONTROL_MAP["exposure"][0]:
            if v4l2_set(dev, n, nudge):
                return v4l2_set(dev, n, int(value))
        return False
    if strategy == "direct":
        auto(1)
        expo()
    elif strategy == "store_then_cycle":
        auto(1)
        expo()
        auto(3)
        time.sleep(0.1)
        auto(1)
    elif strategy == "write_in_auto":
        auto(3)
        expo()
        auto(1)
    else:
        auto(3)
        time.sleep(float(delay_s))
        auto(1)
        expo()
        auto(3)
        time.sleep(0.1)
        auto(1)
        time.sleep(0.1)
        expo()


UVC_UNIT_MS = 0.1        # UVC standard exposure_absolute unit (100 us); not all cameras follow it


def exposure_unit_ms(cfg: Dict[str, Any]) -> Optional[float]:
    """Duration of one exposure_absolute step [ms], measured by `scintcam diagnose` (from the frame rate); None
    without a profile that has it: the UVC standard 0.1 ms is NOT assumed (the DV20 webcam uses ~1.32 us), so
    exposures are then shown and logged in camera steps only."""
    u = ((cfg.get("_profile") or {}).get("exposure") or {}).get("unit_ms_measured")
    return float(u) if isinstance(u, (int, float)) and u > 0 else None


def exposure_text(cfg: Dict[str, Any], steps) -> str:
    """'<ms> ms (<steps> steps)' with a measured unit, '<steps> steps' without."""
    u = exposure_unit_ms(cfg)
    return f"{u * steps:.4g} ms ({steps} steps)" if u else f"{steps} steps (unit not measured)"


def exposure_strategy(cfg: Dict[str, Any]) -> str:
    """From the camera profile when there is one, otherwise the safe full sequence (or direct if cycling is off)."""
    prof = cfg.get("_profile") or {}
    st = (prof.get("exposure") or {}).get("apply")
    if st in EXPOSURE_STRATEGIES:
        return st
    return "full_sequence" if cfg["camera"]["exposure_cycle"] else "direct"


def _lock_v4l2(cam: Dict[str, Any], ctl: Dict[str, Any], strategy: str = "full_sequence",
               unit_ms: Optional[float] = None) -> set:
    dev = cam["device"]
    manual_exposure = ctl.get("auto_exposure") is False and ctl.get("exposure") is not None
    applied, failed, done = [], [], set()
    order = [k for k in SWITCHES if k in ctl] + [k for k in ctl if k not in SWITCHES]
    for key in order:
        if key == "exposure" and (ctl.get("auto_exposure") is True or manual_exposure):
            continue              # auto: ignored (inactive); manual: written by apply_exposure below
        names = CONTROL_MAP[key][0] if key in CONTROL_MAP else [key]   # unknown keys: raw V4L2 control names
        val = _v4l2_value(key, ctl[key])
        if any(v4l2_set(dev, n, val) for n in names):
            applied.append(f"{key}={val}")
            done.add(key)
        else:
            failed.append(key)
    if manual_exposure:
        apply_exposure(dev, _v4l2_value("exposure", ctl["exposure"]), strategy, cam["exposure_cycle_delay_s"])
        applied.append(f"exposure={ctl['exposure']} ({strategy})")
        done.add("exposure")
    if applied:
        log.info("camera controls set: %s", ", ".join(applied))
    if failed:
        log.info("not available on this camera (nothing to pin): %s", ", ".join(failed))
    if manual_exposure and unit_ms:              # the check needs the real unit
        _verify_exposure(dev, ctl["exposure"], unit_ms)
    return set(ctl)        # v4l2-ctl is authoritative: no OpenCV retry, even for controls the camera lacks


def _verify_exposure(dev, exposure, unit_ms: float = UVC_UNIT_MS) -> None:
    """A long exposure forces the sensor to slow down; if it does not fit in the frame period and the driver rate did
    not drop, the camera is ignoring the value (reading it back would not tell)."""
    fps = v4l2_driver_fps(dev)
    exp_ms = unit_ms * float(exposure)
    if fps:
        log.info("exposure %s (%.1f ms); driver reports %.1f fps", exposure, exp_ms, fps)
        if exp_ms > 1150.0 / fps:
            log.warning("%.1f ms does not fit in a %.1f ms frame but the rate did not drop: the camera ignores the "
                        "manual exposure (raise exposure_cycle_delay_s, or lower camera.fps)", exp_ms, 1000.0 / fps)


def _lock_opencv(cap, ctl: Dict[str, Any], skip: set) -> None:
    for key in [k for k in SWITCHES if k in ctl] + [k for k in ctl if k not in SWITCHES]:
        if key in skip or key not in CONTROL_MAP or CONTROL_MAP[key][1] is None:
            continue
        val = ctl[key]
        if key == "auto_exposure":    # backend-specific encodings of "manual"
            val = (1 if IS_LINUX else 0.25) if val is False else (3 if IS_LINUX else 0.75)
        cap.set(CONTROL_MAP[key][1], float(val))
        got = cap.get(CONTROL_MAP[key][1])
        if abs(got - float(val)) > max(1e-3, 0.02 * abs(float(val))):
            log.warning("%s: requested %s, camera reports %s", key, val, got)


# --------------------------------------------------------------------------------------------------------------
# Read-back
# --------------------------------------------------------------------------------------------------------------
_SNAP_PROPS = ("exposure", "auto_exposure", "gain", "brightness", "contrast", "saturation", "gamma", "sharpness",
               "wb_temperature")


def snapshot(cfg: Dict[str, Any], cap=None) -> Dict[str, Any]:
    """State the camera is ACTUALLY in, read back from the driver (not what the configuration asked for).

    exposure_ms = u * exposure_absolute, u = exposure unit [ms] measured in the camera profile; empty without a measured
    unit (the UVC standard 0.1 ms is not assumed), and filled only in manual mode (exposure_auto = 1), because in auto
    mode the driver keeps a stale value flagged 'inactive'.
    """
    dev = cfg["camera"]["device"]
    snap: Dict[str, Any] = {"time": datetime.datetime.now().isoformat(timespec="milliseconds"),
                            "device": v4l2_device(dev) if IS_LINUX else dev}
    ctr = None
    if have_v4l2ctl():
        ctr = v4l2_controls(dev)
        snap.update(controls=ctr, driver_fps=v4l2_driver_fps(dev), format=v4l2_format(dev))
    elif cap is not None:   # other platforms: OpenCV read-back, units are backend-specific
        snap["opencv"] = {k: float(cap.get(CONTROL_MAP[k][1])) for k in _SNAP_PROPS}
    summ: Dict[str, Any] = {"exposure_mode": None, "exposure_absolute": None, "exposure_ms": None,
                            "exposure_unit_ms": exposure_unit_ms(cfg), "gain": None,
                            "driver_fps": snap.get("driver_fps")}
    if ctr:
        ae = ctr.get("exposure_auto") or ctr.get("auto_exposure") or {}
        ea = ctr.get("exposure_absolute") or ctr.get("exposure_time_absolute") or {}
        if "value" in ae:
            summ["exposure_mode"] = "manual" if ae["value"] == 1 else "auto"
        if isinstance(ea.get("value"), int):
            summ["exposure_absolute"] = ea["value"]
            if summ["exposure_mode"] == "manual":
                u = exposure_unit_ms(cfg)
                summ["exposure_ms"] = u * ea["value"] if u else None
        if isinstance(ctr.get("gain", {}).get("value"), int):
            summ["gain"] = ctr["gain"]["value"]
    snap["summary"] = summ
    snap["requested"] = cfg_controls(cfg) if cfg["camera"]["lock"] else None
    prof = cfg.get("_profile")
    if prof:
        snap["profile"] = {"file": cfg["camera"].get("profile"), "created": prof.get("created"),
                           "card": (prof.get("device") or {}).get("card"),
                           "exposure_apply": (prof.get("exposure") or {}).get("apply")}
    return snap


def marker_fields(snap: Optional[dict], stab: Optional[dict], locked: bool) -> Dict[str, Any]:
    """Flat camera keys for the <stem>.done marker, so an analysis can read them without the settings file.

    Every V4L2 control at the start of the clip is copied under its driver name (raw driver value), plus:
      exposure        = exposure_absolute, raw camera units; only in manual mode,
                        because in auto mode the stored value is stale ('inactive') and not the integration time used
      exposure_ms     = exposure x exposure_unit_ms: integration time [ms]
      exposure_unit_ms = duration of one exposure unit [ms], measured by scintcam diagnose (camera profile); both keys
                        are absent without a profile (the UVC standard 0.1 ms is not assumed)
      format          = "<pixel format> <width>x<height>" streamed by the driver
      camera_locked   = whether scintcam pinned the controls (camera.lock)
      stab_drift_pct, stab_rms_pct = last stability check, only when it could judge (lit scene)
    """
    out: Dict[str, Any] = {"camera_locked": bool(locked)}
    ctr = (snap or {}).get("controls") or {}
    for name, d in ctr.items():
        if "value" in d:
            out[name] = d["value"]
    summ = (snap or {}).get("summary", {})
    if summ.get("exposure_mode") == "manual":
        out["exposure"] = summ["exposure_absolute"]
        if summ.get("exposure_ms") is not None:
            out["exposure_ms"] = round(summ["exposure_ms"], 4)
            out["exposure_unit_ms"] = summ["exposure_unit_ms"]
    fmt = (snap or {}).get("format") or {}
    if fmt.get("pixel_format") and fmt.get("width"):
        out["format"] = f"{fmt['pixel_format']} {fmt['width']}x{fmt['height']}"
    if stab and stab.get("stable") is not None:
        out["stab_drift_pct"] = round(stab["drift_pct"], 3)
        out["stab_rms_pct"] = round(stab["rms_pct"], 3)
    return out


def changes(a: Optional[dict], b: Optional[dict]) -> Dict[str, list]:
    """Controls whose value differs between two snapshots: {name: [before, after]}."""
    ca, cb = (a or {}).get("controls") or {}, (b or {}).get("controls") or {}
    return {k: [ca.get(k, {}).get("value"), cb.get(k, {}).get("value")] for k in sorted(set(ca) | set(cb))
            if ca.get(k, {}).get("value") != cb.get(k, {}).get("value")}


def stability(levels: List[float]) -> Optional[Dict[str, Any]]:
    """Level drift on a static scene: the direct test that no automatic control is still adjusting.

    drift_pct = 100 (mean of last 5 - mean of first 5) / mean of first 5 ;  rms_pct = 100 std / mean
    """
    v = np.asarray(levels, float)
    if v.size < 5:
        return None
    first, last = v[:5].mean(), v[-5:].mean()
    res = {"frames": int(v.size), "mean": float(v.mean()), "drift_pct": float(100 * (last - first) / max(first, 1e-6)),
           "rms_pct": float(100 * v.std() / max(v.mean(), 1e-6)),
           "time": datetime.datetime.now().isoformat(timespec="seconds")}
    res["std"] = float(v.std())          # frame-to-frame scatter of the mean level [counts]
    # In darkness the mean is only the black level (a few counts), so a fraction of a count reads as many percent:
    # the test needs a lit, static scene. Below MIN_LEVEL counts it is not judged (stable = None).
    res["stable"] = None if res["mean"] < MIN_LEVEL else (abs(res["drift_pct"]) <= 1.0 and res["rms_pct"] <= 1.0)
    return res


MIN_LEVEL = 20.0       # mean level [counts] below which the stability check cannot judge
