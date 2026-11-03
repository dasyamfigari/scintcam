"""Capture engine: read -> rectify -> record in a background thread, independent of the display.

In a single display loop every redraw (colour map, profiles, readout, a slow remote X display) delays the next
camera read, so the recorded frame rate drops and jitters. Here the capture thread only does what each frame needs
(read, rectify, write, two sums for the seismograph, optional frame accumulation) and publishes the latest frame;
the GUI renders whatever is newest at its own pace and never blocks capture. Controls arrive through a command
queue, so the recorder is only ever touched by the capture thread.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import Geometry, select_channel
from .recorder import Recorder

log = logging.getLogger("scintcam")


@dataclass
class Packet:
    index: int            # frames read since start
    t: float              # host monotonic time [s]
    raw: np.ndarray       # camera frame (BGR)
    rect: Optional[np.ndarray]
    ch: np.ndarray        # analysis channel of rect (or of raw while no geometry is set)


class Seismo:
    """Fixed-size ring buffer of (t, I_tot, n_saturated), keyed by time so the window stays its length in seconds."""

    def __init__(self, seconds: float, fps_max: float = 60.0):
        n = int(seconds * fps_max) + 100
        self.t, self.I, self.s = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
        self.n, self.head, self.count = n, 0, 0
        self.marks: List[Tuple[float, str]] = []
        self.lock = threading.Lock()

    def push(self, t: float, I: float, s: int) -> None:
        with self.lock:
            self.t[self.head], self.I[self.head], self.s[self.head] = t, I, s
            self.head = (self.head + 1) % self.n
            self.count = min(self.count + 1, self.n)

    def mark(self, t: float, label: str) -> None:
        with self.lock:
            self.marks = [m for m in self.marks if m[0] > t - 3600] + [(t, label)]

    def clear(self) -> None:
        with self.lock:
            self.head = self.count = 0
            self.marks = []

    def data(self):
        with self.lock:
            idx = np.arange(self.count) if self.count < self.n else np.r_[self.head:self.n, 0:self.head]
            return self.t[idx].copy(), self.I[idx].copy(), self.s[idx].copy(), list(self.marks)


class CaptureEngine(threading.Thread):
    def __init__(self, cap, cfg: Dict[str, Any], cam_size, nominal_fps: float):
        super().__init__(name="capture", daemon=True)
        self.cap, self.cfg, self.cam_size, self.nominal_fps = cap, cfg, cam_size, nominal_fps
        self.geom: Optional[Geometry] = None
        self.recorder = Recorder(cfg, cam_size, cap)
        self.seismo = Seismo(max(cfg["seismograph"]["minutes"], 0.1) * 60.0)
        self.channel = cfg["rectify"]["channel"]
        self.sat_level = float(cfg["rectify"]["sat_level"] or 0)
        self.stability = None                       # last stability-check result
        self.error: Optional[str] = None
        self._latest: Optional[Packet] = None
        self._lock = threading.Lock()
        self._cmd: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._running = threading.Event()
        self._times: "collections.deque[float]" = collections.deque(maxlen=120)
        self.frames = 0
        self.failed_reads = 0
        # frame averaging for the readout (every frame contributes, not only displayed ones)
        self._avg_n, self._acc, self._acc_k, self.avg_mean, self.avg_serial = 1, None, 0, None, 0
        self._levels: Optional[Tuple[int, List[float], Callable]] = None

    # --------------------------------------------------------------------------------------------- public (GUI thread)
    def latest(self) -> Optional[Packet]:
        with self._lock:
            return self._latest

    @property
    def capture_fps(self) -> float:
        """Rate over the last <=120 frames: (n - 1) / (t_last - t_first)."""
        ts = list(self._times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    def command(self, name: str, arg: Any = None) -> None:
        self._cmd.put((name, arg))

    def set_geometry(self, geom: Optional[Geometry]) -> None:
        self.command("geometry", geom)

    def record(self, on: bool) -> None:
        self.command("record", on)

    def set_average(self, n: int) -> None:
        self.command("average", int(max(1, n)))

    def measure_stability(self, n: int, callback: Callable[[Optional[dict]], None]) -> None:
        """Mean raw-channel level over the next n frames -> camera.stability(); callback runs in the capture thread."""
        self.command("levels", (int(n), callback))

    def stop(self) -> None:
        self._running.clear()

    # ------------------------------------------------------------------------------------------------- capture thread
    def run(self) -> None:
        from .camera import stability
        self._running.set()
        t_dead = None
        while self._running.is_set():
            self._commands()
            ok, raw = self.cap.read()
            t = time.monotonic()
            if not ok or raw is None:
                self.failed_reads += 1
                t_dead = t_dead or t
                if t - t_dead > 5 and self.error is None:
                    self.error = f"no frames for {t - t_dead:.0f} s"
                    log.error("camera delivers no frames (%s); check the cable/USB, then restart", self.error)
                time.sleep(0.01)
                continue
            if t_dead is not None:
                log.info("camera stream recovered after %.1f s", t - t_dead)
                t_dead, self.error = None, None
            t_drv = self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            self._times.append(t)
            geom = self.geom
            rect = geom.rectify(raw) if geom is not None else None
            ch = select_channel(rect if rect is not None else raw, self.channel)
            if geom is not None:   # I_tot = sum of the channel over the rectified frame (rectify_pick raw_intensity)
                self.seismo.push(t, float(cv2.sumElems(ch)[0]),
                                 int(cv2.countNonZero(cv2.compare(ch, self.sat_level, cv2.CMP_GE))) if self.sat_level else 0)
            self.recorder.write(raw, rect, t, t_drv)
            if self._avg_n > 1 and geom is not None:
                self._accumulate(ch)
            if self._levels is not None:
                n, vals, cb = self._levels
                vals.append(float(select_channel(raw, self.channel).mean()))
                if len(vals) >= n:
                    self._levels = None
                    self.stability = stability(vals)
                    cb(self.stability)
            self.frames += 1
            with self._lock:
                self._latest = Packet(self.frames, t, raw, rect, ch)
        if self.recorder.active:
            self.recorder.stop()

    def _accumulate(self, ch: np.ndarray) -> None:
        if self._acc is None or self._acc.shape != ch.shape:
            self._acc, self._acc_k = np.zeros(ch.shape, np.float32), 0
        cv2.accumulate(ch, self._acc)
        self._acc_k += 1
        if self._acc_k >= self._avg_n:
            self.avg_mean, self.avg_serial = self._acc / self._acc_k, self.avg_serial + 1
            self._acc, self._acc_k = None, 0

    def _commands(self) -> None:
        while True:
            try:
                name, arg = self._cmd.get_nowait()
            except queue.Empty:
                return
            try:
                if name == "geometry":
                    self.geom, self._acc, self.avg_mean = arg, None, None
                elif name == "record":
                    if arg and not self.recorder.active:
                        mode = self.cfg["recording"]["fps_mode"]
                        fps = self.capture_fps if (mode == "measured" and len(self._times) >= 30) else self.nominal_fps
                        if mode == "measured" and len(self._times) < 30:
                            log.warning("capture rate not settled yet; header uses the nominal %.1f fps", fps)
                        if self.recorder.start(self.geom, round(fps, 3), self.nominal_fps, self.stability):
                            self.seismo.mark(time.monotonic(), "REC")
                    elif not arg and self.recorder.active:
                        self.recorder.stop()
                        self.seismo.mark(time.monotonic(), "STOP")
                elif name == "average":
                    self._avg_n, self._acc, self.avg_mean = arg, None, None
                elif name == "levels":
                    self._levels = (arg[0], [], arg[1])
                elif name == "clear_seismo":
                    self.seismo.clear()
            except Exception:          # never let a command kill the capture thread
                log.exception("command %s failed", name)
