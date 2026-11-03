"""Two-window GUI: VIEW (rectified image, profiles, seismograph) and CONTROLS (recording, readout, display, camera, log).

Each is its own top-level window, so on a dual monitor (or in a single-application X2Go session) they can sit on
different screens; their positions are remembered between runs. Capture runs in CaptureEngine's thread; this module
only renders the newest frame at display.display_fps and never blocks the capture.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .. import __version__, camera, geometry
from ..analysis import ValuesAverager, beam_readout, profiles
from ..engine import CaptureEngine
from ..render import COLORMAPS, colormap_lut, draw_grid, seismograph, view_image, with_profiles
from .imaging import fit, to_photo
from .picker import Picker

log = logging.getLogger("scintcam")
VIEWS = ("heat", "gray", "color", "raw")
AVG_STEPS = (1, 2, 4, 8, 16, 25, 32, 50, 64, 100, 128, 256)
KEYS = ("space/v record   c/C colormap   m view   g grid   b profiles   h seismograph   e clear seismograph\n"
        "i readout   +/- averaging   s snapshot   x stability   r re-pick   q quit")


class TkLogHandler(logging.Handler):
    """Log records -> queue; the GUI thread drains it (Tk must only be touched from its own thread)."""

    def __init__(self):
        super().__init__()
        self.q: "queue.Queue[logging.LogRecord]" = queue.Queue()
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.q.put(record)


class App:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.dc, self.rc = cfg["display"], cfg["readout"]
        self.root = tk.Tk()
        self.root.title(f"scintcam {__version__} - controls")
        # Buttons never keep the keyboard focus: otherwise [space] would both press the focused button and toggle
        # the recording. Entries still get focus when clicked.
        self.root.option_add("*takeFocus", 0)
        self.view = tk.Toplevel(self.root)
        self.view.title("scintcam - view")
        self.view.configure(bg="#121212")
        self.loghandler = TkLogHandler()
        logging.getLogger("scintcam").addHandler(self.loghandler)

        self.cap = self.engine = None
        self.cam_size = None
        self.geom: Optional[geometry.Geometry] = None
        self.picker: Optional[Picker] = None
        self._open_result = None
        self._last_index = -1
        self._tick_n = 0
        self.readout = None
        self._avg_serial = -1
        self._values = None
        self.cam_snap = None
        self._lut_name, self._lut = None, None
        self._closing = False

        self._vars()
        self._build_view()
        self._build_controls()
        self._bind_keys(self.root)
        self._bind_keys(self.view)
        self._restore_geometry()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.view.protocol("WM_DELETE_WINDOW", self.quit)
        threading.Thread(target=self._open_camera, daemon=True).start()
        self.root.after(50, self._poll)

    # =============================================================================================== state variables
    def _vars(self):
        V = tk.StringVar
        self.v_view = V(value=self.dc["view"])
        self.v_cmap = V(value=self.dc["colormap"])
        self.v_grid = tk.BooleanVar(value=self.dc["grid"])
        self.v_prof = tk.BooleanVar(value=self.dc["profiles"])
        self.v_cut = V(value=self.dc["profile_cut"])
        self.v_seis = tk.BooleanVar(value=self.cfg["seismograph"]["enabled"])
        self.v_seis_min = tk.DoubleVar(value=self.cfg["seismograph"]["minutes"])
        self.v_ro = tk.BooleanVar(value=self.rc["enabled"])
        self.v_avg = V(value=str(self.rc["avg_frames"]))
        self.v_avgmode = V(value=self.rc["avg_mode"])
        self.v_hook = tk.BooleanVar(value=bool(self.cfg["hooks"]["enabled"] and self.cfg["hooks"]["on_recording_done"]))
        for v in (self.v_avg, self.v_avgmode):
            v.trace_add("write", lambda *a: self._apply_avg())

    # ====================================================================================================== windows
    def _build_view(self):
        self.img_label = tk.Label(self.view, bg="#121212", text="opening camera ...", fg="#aaa")
        self.img_label.pack(fill="both", expand=True)
        self.seis_label = tk.Label(self.view, bg="#121212")
        self.seis_label.pack(fill="x")

    def _build_controls(self):
        r = self.root
        main = ttk.Frame(r, padding=8)
        main.pack(fill="both", expand=True)
        big = ("TkDefaultFont", max(10, int(self.rc["font_size"])))
        bold = ("TkDefaultFont", max(10, int(self.rc["font_size"])), "bold")

        # --- recording ------------------------------------------------------------------------------------------
        fr = ttk.LabelFrame(main, text="Recording", padding=6)
        fr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.rec_btn = tk.Button(fr, text="●  RECORD  [space]", font=bold, width=18, bg="#2e7d32", fg="white",
                                 activebackground="#388e3c", command=self.toggle_record, state="disabled")
        self.rec_btn.grid(row=0, column=0, rowspan=3, padx=(0, 10), sticky="ns")
        self.lbl_rec = ttk.Label(fr, text="idle", font=bold)
        self.lbl_rec.grid(row=0, column=1, sticky="w")
        self.lbl_rate = ttk.Label(fr, text="")
        self.lbl_rate.grid(row=1, column=1, sticky="w")
        self.lbl_last = ttk.Label(fr, text="", foreground="#555")
        self.lbl_last.grid(row=2, column=1, sticky="w")
        hk = self.cfg["hooks"]["on_recording_done"]
        cb = ttk.Checkbutton(fr, text="run hook after recording" + (f": {' '.join(hk)[:60]}" if hk else " (none set)"),
                             variable=self.v_hook, command=self._apply_hook)
        cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        if not hk:
            cb.state(["disabled"])
        ttk.Button(fr, text="Snapshot PNG  [s]", command=self.snapshot).grid(row=0, column=2, padx=(10, 0))
        ttk.Label(fr, text=f"→ {os.path.abspath(self.cfg['recording']['dir'])}", foreground="#555").grid(
            row=3, column=2, sticky="e")
        fr.columnconfigure(1, weight=1)

        # --- readout --------------------------------------------------------------------------------------------
        fr = ttk.LabelFrame(main, text="Beam readout (alignment only)", padding=6)
        fr.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.ro_labels = {}
        for i, (k, name) in enumerate((("peak", "peak"), ("n_sat", "saturated px"), ("xbar", "centre x"),
                                       ("ybar", "centre y"), ("fwhm_x", "FWHM x"), ("fwhm_y", "FWHM y"))):
            ttk.Label(fr, text=name, font=big).grid(row=i, column=0, sticky="w")
            self.ro_labels[k] = ttk.Label(fr, text="-", font=bold, width=18, anchor="e")
            self.ro_labels[k].grid(row=i, column=1, sticky="e")
        row = ttk.Frame(fr)
        row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(row, text="on [i]", variable=self.v_ro).pack(side="left")
        ttk.Label(row, text="  average").pack(side="left")
        ttk.Combobox(row, textvariable=self.v_avg, values=[str(a) for a in AVG_STEPS], width=4).pack(side="left")
        ttk.Combobox(row, textvariable=self.v_avgmode, values=["frames", "values"], width=7, state="readonly").pack(side="left")
        self.lbl_avg = ttk.Label(fr, text="", foreground="#555")
        self.lbl_avg.grid(row=7, column=0, columnspan=2, sticky="w")

        # --- display --------------------------------------------------------------------------------------------
        fr = ttk.LabelFrame(main, text="Display", padding=6)
        fr.grid(row=1, column=1, sticky="nsew")
        row = ttk.Frame(fr)
        row.pack(fill="x")
        for v in VIEWS:
            ttk.Radiobutton(row, text=v, value=v, variable=self.v_view).pack(side="left")
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="colormap").pack(side="left")
        ttk.Button(row, text="◀", width=2, command=lambda: self.cycle_cmap(-1)).pack(side="left", padx=(4, 0))
        ttk.Combobox(row, textvariable=self.v_cmap, values=list(COLORMAPS), width=9, state="readonly").pack(side="left")
        ttk.Button(row, text="▶", width=2, command=lambda: self.cycle_cmap(1)).pack(side="left")
        row = ttk.Frame(fr)
        row.pack(fill="x")
        ttk.Checkbutton(row, text="grid [g]", variable=self.v_grid).pack(side="left")
        ttk.Checkbutton(row, text="profiles [b]", variable=self.v_prof).pack(side="left")
        ttk.Combobox(row, textvariable=self.v_cut, values=["peak", "sum"], width=5, state="readonly").pack(side="left")
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(row, text="seismograph [h]", variable=self.v_seis).pack(side="left")
        ttk.Spinbox(row, from_=0.5, to=max(0.5, self.cfg["seismograph"]["minutes"]), increment=0.5,
                    textvariable=self.v_seis_min, width=4).pack(side="left")
        ttk.Label(row, text="min").pack(side="left")
        ttk.Button(row, text="Clear [e]", command=self.clear_seismo).pack(side="left", padx=(6, 0))

        # --- camera ---------------------------------------------------------------------------------------------
        fr = ttk.LabelFrame(main, text="Camera", padding=6)
        fr.grid(row=2, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        self.lbl_cam = ttk.Label(fr, text="opening ...", justify="left")
        self.lbl_cam.pack(anchor="w")
        self.lbl_stab = ttk.Label(fr, text="", justify="left")
        self.lbl_stab.pack(anchor="w", pady=(4, 0))
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Read state", command=self.read_camera).pack(side="left")
        ttk.Button(row, text="Re-apply lock", command=self.relock).pack(side="left", padx=4)
        ttk.Button(row, text="Stability [x]", command=self.stability).pack(side="left")

        # --- geometry -------------------------------------------------------------------------------------------
        fr = ttk.LabelFrame(main, text="Rectification", padding=6)
        fr.grid(row=2, column=1, sticky="nsew", pady=(6, 0))
        self.lbl_geom = ttk.Label(fr, text="not set", justify="left")
        self.lbl_geom.pack(anchor="w")
        ttk.Button(fr, text="Re-pick corners  [r]", command=self.repick).pack(anchor="w", pady=(6, 0))

        # --- log + keys -----------------------------------------------------------------------------------------
        self.logbox = ScrolledText(main, height=9, width=96, font=("TkFixedFont", 9), state="disabled", wrap="word")
        self.logbox.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self.logbox.tag_configure("WARNING", foreground="#b26a00")
        self.logbox.tag_configure("ERROR", foreground="#c62828")
        row = ttk.Frame(main)
        row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(row, text=KEYS, foreground="#555", justify="left").pack(side="left")
        ttk.Button(row, text="Quit", command=self.quit).pack(side="right")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(3, weight=1)

    def _refocus(self, e=None):
        """After a combobox choice, give the keys back to the window (a focused combobox swallows them)."""
        e.widget.selection_clear() if e is not None else None
        e.widget.winfo_toplevel().focus_set()

    def _bind_keys(self, w):
        w.bind_class("TCombobox", "<<ComboboxSelected>>", self._refocus, add="+")
        acts = {"space": self.toggle_record, "v": self.toggle_record, "c": lambda: self.cycle_cmap(1),
                "C": lambda: self.cycle_cmap(-1), "m": self.cycle_view, "g": lambda: self._flip(self.v_grid),
                "b": lambda: self._flip(self.v_prof), "h": lambda: self._flip(self.v_seis), "e": self.clear_seismo,
                "i": lambda: self._flip(self.v_ro), "plus": lambda: self._step_avg(1), "equal": lambda: self._step_avg(1),
                "minus": lambda: self._step_avg(-1), "s": self.snapshot, "x": self.stability, "r": self.repick,
                "q": self.quit}
        for key, fn in acts.items():
            w.bind(f"<KeyPress-{key}>", lambda e, fn=fn: None if self._typing(e) else (fn(), "break")[1])

    @staticmethod
    def _typing(e) -> bool:
        return e.widget.winfo_class() in ("TEntry", "Entry", "TSpinbox", "Spinbox", "TCombobox", "Text")

    def _restore_geometry(self):
        g = self.cfg["gui"]
        saved = {}
        if g["state_file"] and os.path.exists(g["state_file"]):
            try:
                with open(g["state_file"]) as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                pass
        if g["controls_geometry"] or saved.get("controls"):
            self.root.geometry(g["controls_geometry"] or saved["controls"])
        if g["view_geometry"] or saved.get("view"):
            self.view.geometry(g["view_geometry"] or saved["view"])
        else:
            self.view.geometry("760x900")

    # ============================================================================================ camera start-up
    def _open_camera(self):
        """Worker thread: opening can take seconds (mode ladder); the GUI stays responsive meanwhile."""
        try:
            cap, size, fps = camera.open_camera(self.cfg["camera"])
            camera.lock_camera(cap, self.cfg)
            if not self.cfg["camera"]["fps"]:       # rate set by the camera (exposure): measure it AFTER the lock
                fps = camera.measure_fps(cap, 30) or fps
                log.info("camera rate after locking the controls: %.2f fps (measured)", fps)
            self._open_result = ("ok", cap, size, fps)
        except Exception as e:      # reported in the GUI
            self._open_result = ("error", str(e))

    def _camera_ready(self, cap, size, fps):
        self.cap, self.cam_size = cap, size
        self.engine = CaptureEngine(cap, self.cfg, size, fps)
        self.engine.start()
        self._apply_avg()
        self.rec_btn.configure(state="normal")
        self.read_camera()
        if self.cfg["camera"]["stability_check"]:
            self.root.after(4000, self.stability)     # let the sensor settle after the format switch and the lock
        rx = self.cfg["rectify"]
        corners, lines = geometry.load_points(rx["points_json"], rx["setup_name"]) if rx["reuse"] else (None, None)
        if corners and not rx["start_with_picker"]:
            try:
                self._set_geometry(geometry.Geometry.build(corners, lines, size, rx["out_size"], rx["margin"],
                                                           rx["target_mm"], rx["radial"]), save=False)
                log.info("using stored geometry '%s' from %s", rx["setup_name"], rx["points_json"])
                return
            except ValueError as e:
                log.error("stored geometry unusable: %s", e)
        self._open_picker(corners, lines)

    def _open_picker(self, corners=None, lines=None):
        if self.picker is not None and self.picker.winfo_exists():
            self.picker.lift()
            return
        if corners is None and self.geom is not None:
            corners, lines = self.geom.corners, self.geom.lines
        self.picker = Picker(self.root, self.engine, self.cfg, self._picked, self._pick_cancelled, corners, lines)

    def _picked(self, g):
        self.picker = None
        self._set_geometry(g, save=True)

    def _pick_cancelled(self):
        self.picker = None
        if self.geom is None:
            log.warning("no geometry: showing the raw camera image; press [r] to pick the corners")

    def _set_geometry(self, g, save: bool):
        self.geom = g
        self.engine.set_geometry(g)
        if save:
            rx = self.cfg["rectify"]
            geometry.save_geometry(rx["points_json"], rx["setup_name"], g, f"camera:{self.cfg['camera']['device']}")
            log.info("geometry saved as '%s' in %s", rx["setup_name"], rx["points_json"])
        k1 = "" if g.dist is None else f"   k1 {g.dist[0]:.4f}"
        self.lbl_geom.configure(text=f"{g.out_size} x {g.out_size} px   {g.mm_per_px:.4f} mm/px\n"
                                     f"target {g.target_mm:g} mm   channel {self.cfg['rectify']['channel']}{k1}")

    # ================================================================================================ periodic poll
    def _poll(self):
        if self._closing:
            return
        self._drain_log()
        if self._open_result is not None:
            res, self._open_result = self._open_result, None
            if res[0] == "ok":
                self._camera_ready(*res[1:])
            else:
                log.error("%s", res[1])
                self.img_label.configure(text="camera not available:\n" + res[1])
                messagebox.showerror("scintcam", res[1], parent=self.root)
        if self.engine is not None:
            try:
                self._render()
            except Exception:           # a display problem must never stop the GUI (capture is unaffected anyway)
                log.exception("display error")
            self._status()
        self.root.after(max(10, int(1000.0 / float(self.dc["display_fps"]))), self._poll)

    def _drain_log(self):
        lines = []
        while True:
            try:
                rec = self.loghandler.q.get_nowait()
            except queue.Empty:
                break
            lines.append((self.loghandler.format(rec), rec.levelname))
        if lines:
            self.logbox.configure(state="normal")
            for txt, lvl in lines:
                self.logbox.insert("end", txt + "\n", lvl)
            self.logbox.see("end")
            self.logbox.configure(state="disabled")

    def _render(self):
        pkt = self.engine.latest()
        if pkt is None or pkt.index == self._last_index:
            return
        self._last_index = pkt.index
        self._tick_n += 1
        geom, view = self.geom, self.v_view.get()
        sat = float(self.cfg["rectify"]["sat_level"] or 0)

        if pkt.rect is None or view == "raw":
            img = pkt.raw.copy()
        else:
            if self._lut_name != self.v_cmap.get():
                self._lut_name, self._lut = self.v_cmap.get(), colormap_lut(self.v_cmap.get())
            img = view_image(pkt.rect, pkt.ch, view, self._lut, sat, self.dc["sat_color"])
            if self.v_grid.get():
                img = draw_grid(img, geom.mm_per_px, float(self.dc["grid_mm"]))

        if pkt.rect is not None and self.v_ro.get():
            self._update_readout(pkt, geom, sat)
        if pkt.rect is not None and view != "raw" and self.v_prof.get():
            ip = self.readout.get("ipeak") if self.readout else None
            px, py = profiles(pkt.ch, self.v_cut.get(), ip)
            full = sat if (self.dc["profile_norm"] == "fixed" and sat) else max(float(px.max()), float(py.max()), 1.0)
            img = with_profiles(img, px, py, int(self.dc["profile_size"]), full, ip)

        seis_h = int(self.cfg["seismograph"]["height"]) if (self.v_seis.get() and geom is not None) else 0
        aw, ah = max(50, self.view.winfo_width()), max(50, self.view.winfo_height() - seis_h)
        size = fit(img.shape[1], img.shape[0], aw, ah)
        self._photo = to_photo(img, size)
        self.img_label.configure(image=self._photo, text="")
        if seis_h and self._tick_n % 2 == 0:
            t, I, s, marks = self.engine.seismo.data()
            try:
                minutes = max(0.1, float(self.v_seis_min.get()))
            except (tk.TclError, ValueError):
                minutes = float(self.cfg["seismograph"]["minutes"])
            sp = seismograph(t, I, s, time.monotonic(), minutes, aw, seis_h, marks, self.cfg["seismograph"]["y_range"])
            self._seis_photo = to_photo(sp)
            self.seis_label.configure(image=self._seis_photo)
        elif not seis_h:
            self.seis_label.configure(image="")

    def _update_readout(self, pkt, geom, sat):
        mode, n = self.v_avgmode.get(), self._avg_n()
        if n > 1 and mode == "frames":
            if self.engine.avg_serial != self._avg_serial and self.engine.avg_mean is not None:
                self._avg_serial = self.engine.avg_serial
                self.readout = beam_readout(self.engine.avg_mean, geom.mm_per_px, sat)
        elif n > 1:
            if self._values is None or self._values.n != n:
                self._values = ValuesAverager(n)
            got = self._values.push(beam_readout(pkt.ch, geom.mm_per_px, sat))
            if got is not None:
                self.readout = got
        elif self._tick_n % max(1, int(self.rc["every"])) == 0:
            self.readout = beam_readout(pkt.ch, geom.mm_per_px, sat)
        r = self.readout
        if r is None:
            return
        dp, dw = int(self.rc["decimals_pos"]), int(self.rc["decimals_fwhm"])

        def fmt(k, d, sign=""):
            v = r.get(k, float("nan"))
            if v != v:
                return "-"
            s = r.get(k + "_std")
            return f"{v:{sign}.{d}f}" + (f" ± {s:.{d}f}" if s else "") + " mm"
        self.ro_labels["peak"].configure(text=f"{r['peak']:.0f}")
        self.ro_labels["n_sat"].configure(text=f"{r['n_sat']:.0f}", foreground="#c62828" if r["n_sat"] else "")
        self.ro_labels["xbar"].configure(text=fmt("xbar", dp, "+"))
        self.ro_labels["ybar"].configure(text=fmt("ybar", dp, "+"))
        self.ro_labels["fwhm_x"].configure(text=fmt("fwhm_x", dw))
        self.ro_labels["fwhm_y"].configure(text=fmt("fwhm_y", dw))

    def _status(self):
        e, rec = self.engine, self.engine.recorder
        if rec.active:
            dt = time.monotonic() - rec.t_start_mono
            self.lbl_rec.configure(text=f"REC  {int(dt) // 60:02d}:{int(dt) % 60:02d}   {rec.n} frames   {rec.stem}",
                                   foreground="#c62828")
            self.rec_btn.configure(text="■  STOP  [space]", bg="#c62828", activebackground="#d32f2f")
        else:
            self.lbl_rec.configure(text="idle" if e.error is None else f"CAMERA: {e.error}",
                                   foreground="" if e.error is None else "#c62828")
            self.rec_btn.configure(text="●  RECORD  [space]", bg="#2e7d32", activebackground="#388e3c")
        hdr = f"   header {rec.header_fps:.2f} fps" if rec.active else ""
        self.lbl_rate.configure(text=f"capture {e.capture_fps:.2f} fps (nominal {e.nominal_fps:.1f}){hdr}")
        if rec.last:
            L = rec.last
            self.lbl_last.configure(text=f"last: {L['stem']}  {L['frames']} frames  {L['duration_s']:.1f} s  "
                                         f"{L['effective_fps']:.2f} fps" + ("  ⚠ see log" if L["note"] else ""))
        n = self._avg_n()
        fps = e.capture_fps or e.nominal_fps
        self.lbl_avg.configure(text=f"window ≈ {n / fps:.2f} s" if n > 1 and fps else "instantaneous")
        if e.stability is not None and self._stab_shown is not e.stability:
            self._show_stability(e.stability)

    _stab_shown = None

    def _show_stability(self, s):
        self._stab_shown = s
        ok = s["stable"]
        self.lbl_stab.configure(text=f"stability ({s['frames']} fr): drift {s['drift_pct']:+.2f}%  rms {s['rms_pct']:.2f}%  "
                                     f"{'OK' if ok else 'NOT STABLE'}", foreground="#2e7d32" if ok else "#c62828")
        if ok:
            why = "stable"
        elif abs(s["drift_pct"]) <= 1.0:     # no trend, only frame-to-frame scatter
            why = ("level jumps from frame to frame without a trend: lamp flicker (exposure not a multiple of 10 ms "
                   "at 50 Hz) or a changing scene")
        else:
            why = "level drifts: an automatic control is probably still active, or the scene changed"
        (log.info if ok else log.warning)("stability check: drift %+.2f%%, rms %.2f%% -> %s", s["drift_pct"],
                                          s["rms_pct"], why)

    # ===================================================================================================== actions
    def toggle_record(self):
        if self.engine is None:
            return
        if not self.engine.recorder.active and self.cfg["recording"]["source"] != "raw" and self.geom is None:
            log.error("pick the corners first (rectified recording needs a geometry)")
            return
        self.engine.record(not self.engine.recorder.active)

    def _apply_hook(self):
        self.cfg["hooks"]["enabled"] = bool(self.v_hook.get())

    def _avg_n(self) -> int:
        try:
            return max(1, int(float(self.v_avg.get())))
        except ValueError:
            return 1

    def _apply_avg(self):
        if self.engine is not None:
            self.engine.set_average(self._avg_n() if self.v_avgmode.get() == "frames" else 1)
        self._values = None
        self.readout = None

    def _step_avg(self, d):
        n = self._avg_n()
        i = min(range(len(AVG_STEPS)), key=lambda j: abs(AVG_STEPS[j] - n))
        self.v_avg.set(str(AVG_STEPS[max(0, min(len(AVG_STEPS) - 1, i + d))]))

    @staticmethod
    def _flip(var):
        var.set(not var.get())

    def cycle_cmap(self, d):
        names = list(COLORMAPS)
        i = names.index(self.v_cmap.get()) if self.v_cmap.get() in names else 0
        self.v_cmap.set(names[(i + d) % len(names)])
        if self.v_view.get() != "heat":
            self.v_view.set("heat")

    def cycle_view(self):
        self.v_view.set(VIEWS[(VIEWS.index(self.v_view.get()) + 1) % len(VIEWS)])

    def clear_seismo(self):
        if self.engine is not None:
            self.engine.command("clear_seismo")
            log.info("seismograph cleared")

    def snapshot(self):
        pkt = self.engine.latest() if self.engine else None
        if pkt is None:
            return
        d = os.path.abspath(self.cfg["recording"]["dir"])
        os.makedirs(d, exist_ok=True)
        stem = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        cv2.imwrite(os.path.join(d, stem + "_snap_raw.png"), pkt.raw)
        if pkt.rect is not None:
            cv2.imwrite(os.path.join(d, stem + "_snap_rect.png"), pkt.rect)
        log.info("snapshot saved: %s_snap_*.png", stem)

    def read_camera(self):
        if self.cfg is None:
            return
        self.cam_snap = camera.snapshot(self.cfg, self.cap)
        s = self.cam_snap["summary"]
        fmt = self.cam_snap.get("format") or {}
        size = f"{fmt.get('width', self.cam_size[0])}x{fmt.get('height', self.cam_size[1])} {fmt.get('pixel_format', '')}"
        exp = (f"{s['exposure_mode']}" + (f", {s['exposure_ms']:.1f} ms" if s["exposure_ms"] is not None else "")
               if s["exposure_mode"] else "n/a (no v4l2-ctl)")
        self.lbl_cam.configure(text=f"device {self.cfg['camera']['device']}   {size}\n"
                                    f"exposure {exp}   gain {s['gain'] if s['gain'] is not None else 'n/a'}\n"
                                    f"driver fps {s['driver_fps'] if s['driver_fps'] else 'n/a'}")

    def relock(self):
        if self.cap is not None:
            camera.lock_camera(self.cap, self.cfg)
            self.read_camera()

    def stability(self):
        if self.engine is not None:
            n = int(self.cfg["camera"]["stability_frames"])
            self.lbl_stab.configure(text=f"stability: measuring {n} frames (keep the scene static) ...", foreground="")
            self.engine.measure_stability(n, lambda res: None)   # result shown by _status (GUI thread)

    def repick(self):
        if self.engine is None:
            return
        if self.engine.recorder.active:
            log.warning("stop the recording before re-picking")
            return
        self._open_picker()

    # ======================================================================================================== quit
    def quit(self):
        if self._closing:
            return
        if self.engine is not None and self.engine.recorder.active:
            if not messagebox.askyesno("scintcam", "A recording is running. Stop it and quit?", parent=self.root):
                return
        self._closing = True
        try:
            g = self.cfg["gui"]["state_file"]
            if g:
                with open(g, "w") as f:
                    json.dump({"controls": self.root.geometry(), "view": self.view.geometry()}, f)
        except OSError:
            pass
        if self.engine is not None:
            self.engine.stop()                      # the capture thread stops a running recording itself
            self.engine.join(timeout=10)
        if self.cap is not None:
            self.cap.release()
        logging.getLogger("scintcam").removeHandler(self.loghandler)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_gui(cfg: Dict[str, Any]) -> None:
    App(cfg).run()
