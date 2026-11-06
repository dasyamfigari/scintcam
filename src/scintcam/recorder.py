"""Recording of raw and/or rectified video, with per-clip metadata.

Files per clip in recording.dir (stem = start time YYYY-MM-DD-HHMMSS):
  <stem>.mkv / <stem>_rect.mkv   raw / rectified video
  <stem>_frames.csv              capture time of every recorded frame (host monotonic clock, and driver timestamp)
  <stem>_settings.json           written at start (exists even after a crash), completed at stop: camera state read
                                 back from the driver at start and stop, stability check, rectification
  <stem>.done                    written LAST, after every other file is closed: the clip is complete
All methods are called from the capture thread only.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import __version__, camera
from .geometry import Geometry, save_geometry

log = logging.getLogger("scintcam")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "tolist"):
        return _jsonable(o.tolist())
    if isinstance(o, float) and o != o:
        return None
    return o


def atomic_json(path: str, obj: Any, indent: Optional[int] = 1) -> None:
    """Write <path>.tmp then rename, so a reader (e.g. a watcher on another machine) never sees a partial file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=indent)
    os.replace(tmp, path)


def open_writer(path: str, fourcc: str, fps: float, size):
    for code in (fourcc, "MJPG", "XVID"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), float(fps), (int(size[0]), int(size[1])), True)
        if w.isOpened():
            if code != fourcc:
                log.warning("codec %s unavailable, recording %s with %s", fourcc, os.path.basename(path), code)
            return w, code
        w.release()
    raise IOError(f"no usable video codec for {path}")


class StreamWriter:
    """One video file encoded in its own thread. Encoding FFV1 costs ~7 ms per 640x480 frame; done in the capture
    thread it would delay the next camera read (measured: 53 -> 28 fps with two streams). OpenCV releases the GIL while
    encoding, so each stream runs in parallel with capture. The queue is bounded: if the disk or CPU cannot keep up,
    capture waits (the driver then drops frames, which the per-frame timestamps show) instead of memory growing."""

    def __init__(self, path: str, fourcc: str, fps: float, size, maxsize: int = 48):
        self.writer, self.codec = open_writer(path, fourcc, fps, size)
        self.q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.waits = 0                      # times the capture thread had to wait for the encoder
        self.t = threading.Thread(target=self._run, name="writer", daemon=True)
        self.t.start()

    def _run(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            self.writer.write(f)
        self.writer.release()

    def write(self, frame) -> None:
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.waits += 1
            self.q.put(frame)

    def close(self) -> None:
        self.q.put(None)
        self.t.join()


def run_hook(template: List[str], fields: Dict[str, str], log_path: str) -> None:
    """Run the post-recording command in the background; output goes to log_path."""
    try:
        cmd = [str(a).format(**fields) for a in template]
    except (KeyError, IndexError) as e:
        log.error("hook: bad placeholder %s in %s", e, template)
        return
    try:
        lf = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=fields.get("dir") or None)
    except OSError as e:
        log.error("hook could not start: %s", e)
        return
    log.info("hook started: %s", " ".join(cmd))

    def reap():
        rc = p.wait()
        lf.close()
        (log.info if rc == 0 else log.error)("hook finished with exit code %d (log %s)", rc, log_path)
    threading.Thread(target=reap, daemon=True).start()


class Recorder:
    def __init__(self, cfg: Dict[str, Any], cam_size, cap=None):
        self.cfg, self.cam_size, self.cap = cfg, tuple(cam_size), cap
        self.active = False
        self.n = 0
        self.stem: Optional[str] = None
        self.paths: Dict[str, str] = {}
        self.t_start_mono = None
        self.last: Optional[Dict[str, Any]] = None      # summary of the last finished clip (for the GUI)
        self._w: Dict[str, Any] = {}
        self._ts = None
        self._t_first = None
        self._settings: Optional[Dict[str, Any]] = None
        self._settings_path: Optional[str] = None

    # ----------------------------------------------------------------------------------------------------------------
    def start(self, geom: Optional[Geometry], header_fps: float, nominal_fps: float, stability=None) -> bool:
        rc = self.cfg["recording"]
        d = os.path.abspath(rc["dir"])
        os.makedirs(d, exist_ok=True)
        self.stem = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        src = rc["source"]
        if src in ("rectified", "both") and geom is None:
            log.error("cannot record rectified video without a geometry; pick the corners first")
            return False
        cam_start = camera.snapshot(self.cfg, self.cap) if rc["write_settings"] else None   # before the first frame
        self.paths, self._w = {}, {}
        try:
            if src in ("raw", "both"):
                self.paths["raw"] = os.path.join(d, self.stem + rc["ext"])
                self._w["raw"] = StreamWriter(self.paths["raw"], rc["fourcc"], header_fps, self.cam_size)
            if src in ("rectified", "both"):
                self.paths["rect"] = os.path.join(d, self.stem + "_rect" + rc["ext"])
                self._w["rect"] = StreamWriter(self.paths["rect"], rc["fourcc"], header_fps, geom.out_wh)
        except IOError as e:
            log.error("%s", e)
            for w in self._w.values():
                w.close()
            return False
        codecs = {k: w.codec for k, w in self._w.items()}
        if rc["write_timestamps"]:
            self.paths["timestamps"] = os.path.join(d, self.stem + "_frames.csv")
            self._ts = open(self.paths["timestamps"], "w")
            self._ts.write("frame,t_s,t_driver_ms\n")
        if "raw" in self.paths and geom is not None:   # register the raw clip so offline tools need no re-picking
            rx = self.cfg["rectify"]
            save_geometry(rx["points_json"], os.path.basename(self.paths["raw"]), geom, self.paths["raw"])
        if cam_start is not None:
            self._settings_path = os.path.join(d, self.stem + "_settings.json")
            self._settings = {
                "stem": self.stem, "status": "recording", "scintcam_version": __version__,
                "files": {k: os.path.basename(v) for k, v in self.paths.items()},
                "recording": {"start_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
                              "source": src, "fourcc": codecs, "fps_mode": rc["fps_mode"], "header_fps": header_fps,
                              "nominal_fps": nominal_fps, "camera_size": list(self.cam_size)},
                "camera": {"at_start": cam_start, "stability_check": stability},
                "rectification": None if geom is None else dict(
                    geom.as_dict(), channel=self.cfg["rectify"]["channel"], sat_level=self.cfg["rectify"]["sat_level"],
                    points_json=os.path.abspath(self.cfg["rectify"]["points_json"]),
                    points_key=os.path.basename(self.paths["raw"]) if "raw" in self.paths else None),
            }
            atomic_json(self._settings_path, self._settings)
            s = cam_start["summary"]
            exp = "" if s["exposure_absolute"] is None or s["exposure_mode"] != "manual" else (
                f"{s['exposure_ms']:.4g} ms ({s['exposure_absolute']} steps)" if s["exposure_ms"] is not None
                else f"{s['exposure_absolute']} steps (unit not measured)")
            log.info("camera at start: exposure %s %s, gain %s, driver %s fps", s["exposure_mode"], exp,
                     s["gain"], s["driver_fps"])
        self.header_fps, self.n, self._t_first, self.active = header_fps, 0, None, True
        self.t_start_mono = time.monotonic()
        log.info("recording %s (%s, header %.2f fps)", self.stem, src, header_fps)
        return True

    def write(self, raw: np.ndarray, rect: Optional[np.ndarray], t_mono: float, t_driver_ms: float) -> None:
        if not self.active:
            return
        if "raw" in self._w:
            self._w["raw"].write(raw)
        if "rect" in self._w and rect is not None:
            self._w["rect"].write(rect)
        if self._ts is not None:
            if self._t_first is None:
                self._t_first = t_mono
            self._ts.write(f"{self.n},{t_mono - self._t_first:.6f},{t_driver_ms:.3f}\n")
        self.n += 1

    def stop(self) -> Optional[Dict[str, Any]]:
        if not self.active:
            return None
        self.active = False
        dt = time.monotonic() - self.t_start_mono
        waits = sum(w.waits for w in self._w.values())
        for w in self._w.values():
            w.close()                       # drains the queue: every frame is on disk after this
        if waits:
            log.warning("the encoder fell behind %d time(s): capture waited for it (check the frame times)", waits)
        if self._ts is not None:
            self._ts.close()
            self._ts = None
        eff = self.n / dt if dt > 0 else 0.0
        log.info("stopped %s: %d frames, %.1f s, effective %.2f fps", self.stem, self.n, dt, eff)
        note = None
        if self.header_fps and abs(eff - self.header_fps) > 0.10 * self.header_fps:
            note = (f"header fps {self.header_fps:.2f} vs effective {eff:.2f}: use {self.stem}_frames.csv for frame "
                    f"times, not the header")
            log.warning(note)
        rc = self.cfg["recording"]
        if self._settings is not None:      # completed BEFORE the marker, so a watcher finds it final
            cam_stop = camera.snapshot(self.cfg, self.cap)
            changed = camera.changes(self._settings["camera"]["at_start"], cam_stop)
            if changed:
                log.warning("camera controls changed during the recording: %s",
                            ", ".join(f"{k} {a}->{b}" for k, (a, b) in changed.items()))
            self._settings["status"] = "complete"
            self._settings["camera"].update(at_stop=cam_stop, changed_during_recording=changed)
            self._settings["recording"].update(stop_time=datetime.datetime.now().isoformat(timespec="milliseconds"),
                                               frames=self.n, duration_s=round(dt, 3), effective_fps=round(eff, 4),
                                               note=note)
            atomic_json(self._settings_path, self._settings)
        videos = [os.path.basename(self.paths[k]) for k in ("raw", "rect") if k in self.paths]
        d = os.path.dirname(next(iter(self.paths.values())))
        if rc["write_done"] and videos:      # "files" lists videos only: watchers pick the clip from it
            atomic_json(os.path.join(d, self.stem + ".done"),
                        {"stem": self.stem, "files": videos, "frames": self.n, "duration_s": round(dt, 2),
                         "effective_fps": round(eff, 3), "header_fps": self.header_fps,
                         "settings": os.path.basename(self._settings_path) if self._settings else None,
                         "timestamps": os.path.basename(self.paths["timestamps"]) if "timestamps" in self.paths else None,
                         **(camera.marker_fields(self._settings["camera"]["at_start"],
                                                 self._settings["camera"]["stability_check"],
                                                 self.cfg["camera"]["lock"]) if self._settings else {})},
                        indent=None)
        self.last = {"stem": self.stem, "frames": self.n, "duration_s": dt, "effective_fps": eff,
                     "paths": dict(self.paths), "settings": self._settings_path, "note": note}
        hk = self.cfg["hooks"]
        if hk["enabled"] and hk["on_recording_done"] and self.n >= 2:
            fields = {"raw": self.paths.get("raw", ""), "rect": self.paths.get("rect", ""),
                      "settings": self._settings_path or "", "timestamps": self.paths.get("timestamps", ""),
                      "stem": self.stem, "dir": d}
            run_hook(hk["on_recording_done"], fields, os.path.join(d, self.stem + "_hook.log"))
        self._settings = self._settings_path = None
        return self.last
