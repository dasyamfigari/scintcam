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
from tkinter import font as tkfont
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
CAM_FIELDS = (("exposure_ms", "exposure (ms)"), ("gain", "gain"), ("gamma", "gamma"),
              ("brightness", "brightness"), ("contrast", "contrast"), ("saturation", "saturation"))
CAM_V4L2 = {"gain": "gain", "gamma": "gamma", "brightness": "brightness", "contrast": "contrast",
            "saturation": "saturation"}            # field -> V4L2 control name
AVG_STEPS = (1, 2, 4, 8, 16, 25, 32, 50, 64, 100, 128, 256)
KEYS = ("space/v record   c/C colormap   m view   g grid   b profiles   h seismograph   e clear seismograph\n"
        "i readout   +/- averaging   [ ] exposure step down/up   s snapshot   x stability   r re-pick   q quit")
READOUT_ROWS = (("peak", "peak"), ("n_sat", "saturated px"), ("xbar", "centroid x"), ("ybar", "centroid y"),
                ("fwhm_x", "FWHM x"), ("fwhm_y", "FWHM y"))
READOUT_FONT_RANGE = (8, 72)    # readout text size limits [pt]


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
        self._cam_busy = False
        self._cam_done = False
        self._ae_read = None          # auto exposure on: latest exposure read from the driver (ms), from a thread
        self._ae_next = 0.0
        self._ae_stale = None         # exposure_absolute stored when auto was switched on (many cameras keep reporting it)
        self._cam_shown = {}
        self.cam_labels = {}          # field -> text last read back from the camera

        self._saved = self._load_state()
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
        size = self._saved.get("readout_font") or int(self.rc["font_size"])
        size = int(min(max(size, READOUT_FONT_RANGE[0]), READOUT_FONT_RANGE[1]))
        self.ro_font = tkfont.nametofont("TkDefaultFont").copy()    # the six readout rows only; +/- below resize them
        self.ro_font.configure(size=size)
        self.ro_font_b = self.ro_font.copy()
        self.ro_font_b.configure(weight="bold")
        for i, (k, name) in enumerate(READOUT_ROWS):
            ttk.Label(fr, text=name, font=self.ro_font).grid(row=i, column=0, sticky="w")
            self.ro_labels[k] = ttk.Label(fr, text="-", font=self.ro_font_b, width=18, anchor="e")
            self.ro_labels[k].grid(row=i, column=1, sticky="e")
        row = ttk.Frame(fr)
        row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(row, text="−", width=2, command=lambda: self.readout_font(-1)).pack(side="right")
        ttk.Button(row, text="+", width=2, command=lambda: self.readout_font(1)).pack(side="right")
        self.lbl_rofont = ttk.Label(row, text=f"text {size} pt", foreground="#555")
        self.lbl_rofont.pack(side="right", padx=(0, 4))
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
        grid = ttk.Frame(fr)
        grid.pack(fill="x", pady=(6, 0))
        self.cam_entries = {}
        for i, (key, label) in enumerate(CAM_FIELDS):
            lb = ttk.Label(grid, text=label)
            lb.grid(row=i // 3 * 2, column=i % 3, sticky="w", padx=(0, 6))
            self.cam_labels[key] = (lb, label)
            v = tk.StringVar()
            e = ttk.Entry(grid, textvariable=v, width=8)
            e.grid(row=i // 3 * 2 + 1, column=i % 3, sticky="w", padx=(0, 6))
            e.bind("<Return>", lambda ev: self.apply_camera())
            self.cam_entries[key] = (v, e)
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="exposure step").pack(side="left")
        self.btn_rung_dn = ttk.Button(row, text="◀ [", width=4, command=lambda: self.step_rung(-1))
        self.btn_rung_dn.pack(side="left", padx=(6, 0))
        self.btn_rung_up = ttk.Button(row, text="] ▶", width=4, command=lambda: self.step_rung(1))
        self.btn_rung_up.pack(side="left")
        self.lbl_rung = ttk.Label(row, text="")
        self.lbl_rung.pack(side="left", padx=(6, 0))
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=(6, 0))
        self.var_ae = tk.BooleanVar(value=False)
        self.chk_ae = ttk.Checkbutton(row, text="auto exposure", variable=self.var_ae, command=self.toggle_auto_exposure)
        self.chk_ae.pack(side="left")
        self.lbl_ae = ttk.Label(row, text="", foreground="#b26a00")
        self.lbl_ae.pack(side="left", padx=(6, 0))
        row = ttk.Frame(fr)
        row.pack(fill="x", pady=(6, 0))
        self.btn_apply = ttk.Button(row, text="Apply", command=self.apply_camera)
        self.btn_apply.pack(side="left")
        self.btn_save = ttk.Button(row, text="Save to config", command=self.save_camera)
        self.btn_save.pack(side="left", padx=4)
        if not self.cfg.get("_path"):
            self.btn_save.state(["disabled"])
        ttk.Button(row, text="Read state", command=self.read_camera).pack(side="left")
        ttk.Button(row, text="Stability [x]", command=self.stability).pack(side="left", padx=4)

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
                "minus": lambda: self._step_avg(-1), "bracketleft": lambda: self.step_rung(-1),
                "bracketright": lambda: self.step_rung(1), "s": self.snapshot, "x": self.stability, "r": self.repick,
                "q": self.quit}
        for key, fn in acts.items():
            w.bind(f"<KeyPress-{key}>", lambda e, fn=fn: None if self._typing(e) else (fn(), "break")[1])

    @staticmethod
    def _typing(e) -> bool:
        return e.widget.winfo_class() in ("TEntry", "Entry", "TSpinbox", "Spinbox", "TCombobox", "Text")

    def _load_state(self) -> Dict[str, Any]:
        """Window positions and readout text size remembered from the last run (gui.state_file)."""
        g = self.cfg["gui"]["state_file"]
        if g and os.path.exists(g):
            try:
                with open(g) as f:
                    d = json.load(f)
                return d if isinstance(d, dict) else {}
            except (OSError, ValueError):
                pass
        return {}

    def readout_font(self, step: int):
        """Text size of the six readout rows (peak, saturated, centroid x/y, FWHM x/y) by `step` points."""
        size = int(min(max(int(self.ro_font.cget("size")) + step, READOUT_FONT_RANGE[0]), READOUT_FONT_RANGE[1]))
        self.ro_font.configure(size=size)
        self.ro_font_b.configure(size=size)
        self.lbl_rofont.configure(text=f"text {size} pt")

    def _restore_geometry(self):
        g = self.cfg["gui"]
        saved = self._saved
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
            camera.load_profile(self.cfg)
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
        self._poll_auto_exposure()
        if self._open_result is not None:
            res, self._open_result = self._open_result, None
            if res[0] == "ok":
                self._camera_ready(*res[1:])
            else:
                log.error("%s", res[1])
                self.img_label.configure(text="camera not available:\n" + res[1])
                messagebox.showerror("scintcam", res[1], parent=self.root)
        if self._cam_done:
            self._cam_done = self._cam_busy = False
            self.read_camera()
            s = self.cam_snap["summary"]
            log.info("camera now: exposure %s%s, gain %s, driver %s fps", s["exposure_mode"],
                     "" if s["exposure_mode"] != "manual" or s["exposure_absolute"] is None
                     else " " + camera.exposure_text(self.cfg, s["exposure_absolute"]), s["gain"], s["driver_fps"])
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
        self.btn_apply.state(["disabled"] if (rec.active or self._cam_busy) else ["!disabled"])
        if (self.cam_snap or {}).get("summary", {}).get("exposure_mode") is not None:
            self.chk_ae.state(["disabled"] if (rec.active or self._cam_busy) else ["!disabled"])
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
        lv = f"level {s['mean']:.1f} ± {s['std']:.2f} counts"
        if ok is None:
            self.lbl_stab.configure(text=f"stability: scene too dark to judge ({lv})", foreground="#b26a00")
            log.info("stability check: scene too dark to judge (%s); run it with the room lit or a steady lamp", lv)
            return
        self.lbl_stab.configure(text=f"stability ({s['frames']} fr): drift {s['drift_pct']:+.2f}%  rms {s['rms_pct']:.2f}%  "
                                     f"{'OK' if ok else 'NOT STABLE'}\n{lv}", foreground="#2e7d32" if ok else "#c62828")
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
        if not self.engine.recorder.active and self.var_ae.get():
            log.warning("recording with AUTO EXPOSURE ON: intensities in this clip are not comparable "
                        "(the exposure in use is logged in the settings file)")
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
        exp = (f"{s['exposure_mode']}" + (f", {self._fmt_exp(s['exposure_absolute'])}"
                                          if s["exposure_mode"] == "manual" and s["exposure_absolute"] is not None else "")
               if s["exposure_mode"] else "n/a (no v4l2-ctl)")
        self._fill_camera_fields()
        self.lbl_cam.configure(text=f"device {self.cfg['camera']['device']}   {size}\n"
                                    f"exposure {exp}   gain {s['gain'] if s['gain'] is not None else 'n/a'}\n"
                                    f"driver fps {s['driver_fps'] if s['driver_fps'] else 'n/a'}")

    def _fill_camera_fields(self):
        """Fields show what the camera really has (driver read-back); controls it lacks are disabled."""
        ctr = (self.cam_snap or {}).get("controls") or {}
        summ = (self.cam_snap or {}).get("summary") or {}
        auto = summ.get("exposure_mode") == "auto"
        self.var_ae.set(auto)
        self.chk_ae.state(["disabled"] if summ.get("exposure_mode") is None else ["!disabled"])
        self.lbl_ae.configure(text="ON: not for measurements" if auto else "")
        for key, (var, entry) in self.cam_entries.items():
            if key == "exposure_ms":
                lb, text = self.cam_labels[key]
                lb.configure(text="exposure (ms)" if camera.exposure_unit_ms(self.cfg) else "exposure (steps)")
                val = None if summ.get("exposure_absolute") is None or summ.get("exposure_mode") != "manual" \
                    else self._unit() * summ["exposure_absolute"]
                if auto:      # shown only once it differs from the stored manual value, i.e. it is really live
                    ea = summ.get("exposure_absolute")
                    val = self._unit() * ea if ea is not None and ea != self._ae_stale else None
                var.set("" if val is None else self._fieldtxt(val))
                self._cam_shown[key] = var.get()
                have = "exposure_absolute" in ctr or "exposure_time_absolute" in ctr
            else:
                d = ctr.get(CAM_V4L2[key], {})
                var.set("" if "value" not in d else str(d["value"]))
                self._cam_shown[key] = var.get()
                have = "value" in d
                lb, text = self.cam_labels[key]
                eff = (self._profile_ctl(CAM_V4L2[key]) or {}).get("effect")
                lb.configure(text=text + (" (no effect)" if eff == "none" else ""),
                             foreground="#888888" if eff == "none" else "")
            entry.state(["!disabled"] if ((have or not ctr) and not (auto and key == "exposure_ms")) else ["disabled"])
        if auto and not self.cam_entries["exposure_ms"][0].get():
            self.lbl_ae.configure(text="ON: not for measurements; camera does not report its exposure")
        self._fill_rung(summ)

    def _fill_rung(self, summ):
        """Exposure-ladder line: rung k of N and its exposure, or where an off-ladder exposure lies."""
        lad = summ.get("exposure_ladder") or []
        ea = summ.get("exposure_absolute")
        ok = bool(lad) and summ.get("exposure_mode") is not None
        for b in (self.btn_rung_dn, self.btn_rung_up):
            b.state(["!disabled"] if ok else ["disabled"])
        if not lad:
            self.lbl_rung.configure(text="(no ladder: no exposure control)", foreground="#888888")
        elif summ.get("exposure_mode") != "manual" or ea is None:
            self.lbl_rung.configure(text=f"auto ({len(lad)} rungs; ◀ ▶ = manual)", foreground="#b26a00")
        elif summ.get("exposure_rung"):
            self.lbl_rung.configure(text=f"{summ['exposure_rung']} / {len(lad)}   {self._fmt_exp(ea)}", foreground="")
        else:
            self.lbl_rung.configure(text=f"off the ladder: {self._fmt_exp(ea)}",
                                    foreground="#b26a00")

    def step_rung(self, d: int):
        """Next exposure rung below (d < 0) or above (d > 0) the current exposure; applied at once."""
        summ = (self.cam_snap or {}).get("summary") or {}
        lad = summ.get("exposure_ladder") or []
        if not lad or self.cap is None:
            return
        ea = summ.get("exposure_absolute") if summ.get("exposure_mode") == "manual" else None
        if ea is None:                                   # auto: start from the middle of the ladder
            tgt = lad[len(lad) // 2]
        elif d > 0:
            tgt = next((v for v in lad if v > ea), None)
        else:
            tgt = next((v for v in reversed(lad) if v < ea), None)
        if tgt is None:
            log.info("exposure already at the %s rung (%s)", "top" if d > 0 else "bottom", self._fmt_exp(ea))
            return
        self._apply_values({"auto_exposure": False, "exposure": int(tgt)},
                           note=f"exposure rung {lad.index(tgt) + 1}/{len(lad)}")

    def _profile_ctl(self, name):
        return ((self.cfg.get("_profile") or {}).get("controls") or {}).get(name)

    def _unit(self):
        """Field unit per camera step: the measured ms per step (camera profile), else 1 (the field is in steps)."""
        return camera.exposure_unit_ms(self.cfg) or 1.0

    def _fieldtxt(self, val):
        """Exposure field text: ms with 5 significant digits, or whole steps without a measured unit."""
        return f"{val:.5g}" if camera.exposure_unit_ms(self.cfg) else str(int(round(val)))

    def _fmt_exp(self, steps):
        return camera.exposure_text(self.cfg, steps)

    def _profile_exp(self):
        return (self.cfg.get("_profile") or {}).get("exposure") or {}

    def _camera_values(self, changed_only=True):
        """Typed values -> {control: value}, checked against the driver's range. Empty field = not changed.
        Fields still showing the read-back value are skipped (Apply) or kept unchecked (Save): some cameras report a
        current value outside their own declared range."""
        ctr = (self.cam_snap or {}).get("controls") or {}
        out = {}
        for key, (var, entry) in self.cam_entries.items():
            txt = var.get().strip().replace(",", ".")
            if not txt or entry.instate(["disabled"]):
                continue
            unchanged = txt == self._cam_shown.get(key, "").replace(",", ".")
            if unchanged and changed_only:
                continue
            try:
                num = float(txt)
            except ValueError:
                raise ValueError(f"{key}: '{txt}' is not a number")
            if key == "exposure_ms":           # ms -> camera steps (unit from the camera profile)
                name, val = "exposure", int(round(num / self._unit()))
                d = ctr.get("exposure_absolute") or ctr.get("exposure_time_absolute") or {}
            else:
                name, val = key, int(round(num))
                d = ctr.get(CAM_V4L2[key], {})
            lo, hi = d.get("min"), d.get("max")
            if not unchanged and isinstance(lo, int) and isinstance(hi, int) and not lo <= val <= hi:
                u = self._unit()
                shown = (f"{lo * u:.4g}-{hi * u:.4g} {'ms' if u != 1.0 else 'steps'}" if key == "exposure_ms"
                         else f"{lo}-{hi}")
                raise ValueError(f"{key}: {num:g} outside the camera's range {shown}; nothing applied")
            out[name] = val
        if "exposure" in out:
            out["auto_exposure"] = False
        return out

    def apply_camera(self):
        if self.cap is None or self._cam_busy:
            return
        if self.engine is not None and self.engine.recorder.active:
            log.warning("camera settings cannot change during a recording; stop it first")
            return
        try:
            vals = self._camera_values()
        except ValueError as e:
            log.error("%s", e)
            return
        if not vals:
            log.info("nothing changed: edit a field, then Apply")
            return
        self._apply_values(vals)

    def _apply_values(self, vals: Dict[str, Any], note: str = ""):
        """Write camera controls in a worker thread (shared by Apply and the exposure-ladder buttons)."""
        if self.cap is None or self._cam_busy:
            return
        if self.engine is not None and self.engine.recorder.active:
            log.warning("camera settings cannot change during a recording; stop it first")
            return
        self.cfg["camera"]["controls"].update(vals)   # the next recordings log these as the requested values
        self._cam_busy = True
        self.btn_apply.state(["disabled"])
        log.info("applying camera settings%s: %s", f" ({note})" if note else "",
                 ", ".join(f"{k}={v}" for k, v in vals.items()))
        ex = self._profile_exp()
        if "exposure" in vals and ex.get("full_rate_max") and vals["exposure"] > ex["full_rate_max"]:
            log.warning("%s is above %s: on this camera the frame rate drops (camera profile)",
                        self._fmt_exp(vals["exposure"]), self._fmt_exp(ex["full_rate_max"]))
        settle = float(ex.get("settle_s") or 0.5)

        def work():        # v4l2-ctl calls and the auto->manual pause take ~0.5 s: off the GUI thread
            try:
                camera.lock_camera(self.cap, self.cfg)
                time.sleep(settle)         # until the change shows in the image
            finally:
                self._cam_done = True
        threading.Thread(target=work, daemon=True).start()

    def _poll_auto_exposure(self):
        """While auto exposure is on, show the exposure the camera is choosing (read about once a second)."""
        if self._ae_read is not None:
            ms, self._ae_read = self._ae_read, None
            if self.var_ae.get() and not self._cam_busy and self._ae_stale is not None \
                    and round(ms / self._unit()) != self._ae_stale:
                txt = self._fieldtxt(ms)
                self.cam_entries["exposure_ms"][0].set(txt)
                self._cam_shown["exposure_ms"] = txt
                self.lbl_ae.configure(text="ON: not for measurements")
        now = time.monotonic()
        if not self.var_ae.get() or self._cam_busy or self.cap is None or now < self._ae_next \
                or self._profile_exp().get("auto_readback") == "stale":
            return
        self._ae_next = now + 1.0
        dev = self.cfg["camera"]["device"]

        def work():
            ctr = camera.v4l2_controls(dev) or {}
            ea = ctr.get("exposure_absolute") or ctr.get("exposure_time_absolute") or {}
            if isinstance(ea.get("value"), int):
                self._ae_read = self._unit() * ea["value"]
            if (ctr.get("exposure_auto") or ctr.get("auto_exposure") or {}).get("value") == 1:
                self._ae_read = None       # switched to manual meanwhile
        threading.Thread(target=work, daemon=True).start()

    def toggle_auto_exposure(self):
        want = self.var_ae.get()
        if self.cap is None or self._cam_busy or (self.engine is not None and self.engine.recorder.active):
            self.var_ae.set(not want)
            if self.engine is not None and self.engine.recorder.active:
                log.warning("camera settings cannot change during a recording; stop it first")
            return
        ctl = self.cfg["camera"]["controls"]
        ctl["auto_exposure"] = want
        ctl.pop("exposure", None)
        if want:
            self._ae_stale = ((self.cam_snap or {}).get("summary") or {}).get("exposure_absolute")
            log.warning("auto exposure ON: the camera changes the exposure with the scene brightness, so intensities "
                        "are NOT comparable between frames or clips. Use it only to find a setting, then switch it off.")
        else:
            log.info("auto exposure OFF: switching to manual and matching the brightness auto exposure had ...")
        self._cam_busy = True
        self.btn_apply.state(["disabled"])

        def work():
            try:
                if want:
                    camera.lock_camera(self.cap, self.cfg)
                else:
                    self._match_auto_level()
            except Exception:
                log.exception("auto exposure switch failed")
            finally:
                self._cam_done = True
        threading.Thread(target=work, daemon=True).start()

    def _level(self, n=8):
        """Mean level of the next n frames (analysis channel of the raw frame), from the capture thread."""
        ev, box = threading.Event(), {}
        self.engine.measure_stability(n, lambda r: (box.update(r=r), ev.set()))
        ev.wait(15)
        return (box.get("r") or {}).get("mean")

    def _match_auto_level(self):
        """Worker: auto -> manual, then set the manual exposure that gives the level auto exposure had.

        Many cameras return to their stored manual exposure on that switch, or keep the auto one without reporting
        it, so the level is matched instead: E_next = E L_auto / L, with E exposure [camera steps], L mean level
        [counts] (level ~ exposure), repeated until |L - L_auto| <= max(2 counts, 3 %). The final E is a written,
        known value, so the clips log the exposure really in use."""
        cam, ex = self.cfg["camera"], self._profile_exp()
        settle = float(ex.get("settle_s") or 1.0)
        target = self._level(10)
        camera.lock_camera(self.cap, self.cfg)          # auto_exposure = false, no exposure: plain switch
        time.sleep(settle)
        lv = self._level(8)
        if target is None or lv is None or not camera.have_v4l2ctl():
            return
        ctr = camera.v4l2_controls(cam["device"]) or {}
        name = next((n for n in camera.CONTROL_MAP["exposure"][0] if n in ctr), None)
        if name is None or ex.get("apply") == "none":
            log.warning("auto exposure OFF: level %.1f (auto gave %.1f); the exposure cannot be set on this camera",
                        lv, target)
            return
        d = ctr[name]
        e, lo, hi = d.get("value"), d.get("min", 1), d.get("max", 100000)
        tol = max(2.0, 0.03 * target)
        strategy = camera.exposure_strategy(self.cfg)
        changed = False
        if ex.get("auto_readback") != "live" and isinstance(e, int):
            # the read-back is the stored number, not necessarily the exposure in use (a camera that keeps the auto
            # exposure still reports the old stored value): write a known exposure, then match from there
            camera.apply_exposure(cam["device"], e, strategy, cam["exposure_cycle_delay_s"])
            time.sleep(settle)
            lv, changed = self._level(8), True
            if lv is None:
                return
        for _ in range(8):
            if abs(lv - target) <= tol or not isinstance(e, int):
                break
            new = e / 2 if lv >= 250 else (e * 4 if lv < 1 else e * target / lv)
            new = int(min(max(round(new), lo), hi))
            if new == e:
                break
            e, changed = new, True
            camera.apply_exposure(cam["device"], e, strategy, cam["exposure_cycle_delay_s"])
            time.sleep(settle)
            lv = self._level(8)
            if lv is None:
                return
        if changed:
            cam["controls"]["exposure"] = e
        log.info("auto exposure OFF: manual %s, level %.1f (auto gave %.1f)",
                 self._fmt_exp(e) if isinstance(e, int) else "exposure", lv, target)
        if abs(lv - target) > tol:
            log.warning("could not match the auto brightness within 3 %%: set exposure (ms) by hand")
        if isinstance(e, int) and ex.get("full_rate_max") and e > ex["full_rate_max"]:
            log.warning("%s is above %s: on this camera the frame rate drops",
                        self._fmt_exp(e), self._fmt_exp(ex["full_rate_max"]))

    def save_camera(self):
        path = self.cfg.get("_path")
        if not path:
            return
        try:
            vals = self._camera_values(changed_only=False)
        except ValueError as e:
            log.error("%s", e)
            return
        if not self.chk_ae.instate(["disabled"]):
            vals["auto_exposure"] = bool(self.var_ae.get())
        if not vals:
            return
        from ..config import save_controls
        try:
            save_controls(path, vals)
        except Exception as e:
            log.error("could not save to %s: %s", path, e)
            return
        log.info("saved to %s [camera.controls]: %s", os.path.basename(path), ", ".join(f"{k}={v}" for k, v in vals.items()))

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
                    json.dump({"controls": self.root.geometry(), "view": self.view.geometry(),
                               "readout_font": int(self.ro_font.cget("size"))}, f)
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
