"""Corner picker window: click / auto-detect the 4 target corners on a frozen live frame, check the rectified preview.

Corner 1 maps to the TOP-LEFT of the rectified image, then clockwise (TR, BR, BL). Click them in the order that gives
the orientation you want. Lines (optional): >= 3 points on physically straight, long edges, for the radial fit.
"""
from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional

import cv2
import numpy as np

from ..detect import detect_target, order_corners
from ..geometry import Geometry
from .imaging import fit, to_photo

log = logging.getLogger("scintcam")

HELP = ("Click 4 corners: #1 becomes the TOP-LEFT of the rectified image, then clockwise.\n"
        "With 4 placed, a click moves the nearest one.  Tab: select corner;  arrows: move it.\n"
        "Lines (optional): >= 3 points on a long straight edge, for radial distortion.")


class Picker(tk.Toplevel):
    def __init__(self, master, engine, cfg, on_done: Callable[[Geometry], None], on_cancel: Callable[[], None],
                 corners=None, lines=None):
        super().__init__(master)
        self.title("scintcam - pick target corners")
        self.engine, self.cfg, self.on_done, self.on_cancel = engine, cfg, on_done, on_cancel
        self.pc = cfg["picker"]
        self.corners: List[List[float]] = [list(map(float, p)) for p in (corners or [])][:4]
        self.lines: List[List[List[float]]] = [[list(map(float, p)) for p in L] for L in (lines or [])] or [[]]
        self.mode = tk.StringVar(value="corners")
        self.sel = len(self.corners) - 1
        self.cursor = None
        self.loupe_on_sel = False
        self.preview_geom: Optional[Geometry] = None
        self.frame = None

        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Label(body, bg="black", cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        side = ttk.Frame(body, padding=(8, 0))
        side.grid(row=0, column=1, sticky="ns")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.pick_panel = ttk.Frame(side)
        self.pick_panel.pack(fill="x")
        b = dict(fill="x", pady=2)
        ttk.Button(self.pick_panel, text="Auto-detect  [a]", command=self.auto).pack(**b)
        ttk.Radiobutton(self.pick_panel, text="Corners  [c]", variable=self.mode, value="corners",
                        command=self.redraw).pack(anchor="w")
        ttk.Radiobutton(self.pick_panel, text="Lines  [l]", variable=self.mode, value="lines",
                        command=self.redraw).pack(anchor="w")
        ttk.Button(self.pick_panel, text="New line  [n]", command=self.new_line).pack(**b)
        ttk.Button(self.pick_panel, text="Undo  [z]", command=self.undo).pack(**b)
        ttk.Button(self.pick_panel, text="Clear all", command=self.clear).pack(**b)
        ttk.Button(self.pick_panel, text="New frame  [f]", command=self.freeze).pack(**b)
        ttk.Separator(self.pick_panel).pack(fill="x", pady=6)
        ttk.Button(self.pick_panel, text="Preview  [Enter]", command=self.preview).pack(**b)
        ttk.Button(self.pick_panel, text="Cancel  [Esc]", command=self.cancel).pack(**b)

        self.prev_panel = ttk.Frame(side)
        ttk.Button(self.prev_panel, text="Use this geometry  [Enter]", command=self.accept).pack(**b)
        ttk.Button(self.prev_panel, text="Back to corners  [Esc]", command=self.back).pack(**b)

        self.status = ttk.Label(side, text="", wraplength=230, justify="left")
        self.status.pack(fill="x", pady=(10, 0))
        ttk.Label(side, text=HELP, wraplength=230, justify="left", foreground="#666").pack(fill="x", pady=(10, 0))

        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<Button-1>", self._click)
        for k, fn in (("a", self.auto), ("c", lambda: self._mode("corners")), ("l", lambda: self._mode("lines")),
                      ("n", self.new_line), ("z", self.undo), ("f", self.freeze)):
            self.bind(f"<Key-{k}>", lambda e, fn=fn: fn())
        for k, d in (("Left", (-1, 0)), ("Right", (1, 0)), ("Up", (0, -1)), ("Down", (0, 1))):
            self.bind(f"<{k}>", lambda e, d=d: self.nudge(*d))
        self.bind("<Tab>", self._tab)
        self.bind("<Return>", lambda e: self.accept() if self.preview_geom else self.preview())
        self.bind("<Escape>", lambda e: self.back() if self.preview_geom else self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.after(50, self.freeze)
        self.focus_force()

    # ------------------------------------------------------------------------------------------------- frame / view
    def freeze(self):
        pkt = self.engine.latest()
        if pkt is None:
            self.after(100, self.freeze)
            return
        self.frame = pkt.raw.copy()
        h, w = self.frame.shape[:2]
        self.disp = fit(w, h, max(w, min(1000, self.winfo_screenwidth() - 320)), min(760, self.winfo_screenheight() - 120))
        self.s = self.disp[0] / float(w)
        self.base = cv2.resize(self.frame, self.disp, interpolation=cv2.INTER_LINEAR)
        self.redraw()

    def _pt(self, p):
        return int(round(p[0] * self.s)), int(round(p[1] * self.s))

    def redraw(self):
        if self.frame is None or self.preview_geom is not None:
            return
        img = self.base.copy()
        for L in self.lines:
            for p in L:
                cv2.circle(img, self._pt(p), 3, (0, 128, 255), -1, cv2.LINE_AA)
            if len(L) > 1:
                cv2.polylines(img, [np.array([self._pt(p) for p in L], np.int32)], False, (0, 128, 255), 1, cv2.LINE_AA)
        if len(self.corners) > 1:
            cv2.polylines(img, [np.array([self._pt(p) for p in self.corners], np.int32)], len(self.corners) == 4,
                          (0, 200, 0), 1, cv2.LINE_AA)
        for i, p in enumerate(self.corners):
            q = self._pt(p)
            cv2.circle(img, q, 5, (0, 255, 0), -1, cv2.LINE_AA)
            if i == self.sel and self.mode.get() == "corners":
                cv2.circle(img, q, 10, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        if self.mode.get() == "corners":
            c = self.corners[self.sel] if (self.loupe_on_sel and 0 <= self.sel < len(self.corners)) else self.cursor
            if c is not None:
                self._loupe(img, c)
        self.photo = to_photo(img)
        self.canvas.configure(image=self.photo)
        n_pts = sum(len(L) for L in self.lines)
        self.status.configure(text=f"{len(self.corners)}/4 corners" + (f", {n_pts} line points" if n_pts else "")
                              + ("" if self.sel < 0 or not self.corners else
                                 f"\nselected #{self.sel + 1}: ({self.corners[self.sel][0]:.2f}, "
                                 f"{self.corners[self.sel][1]:.2f})"))

    def _loupe(self, img, c):
        """Magnified CLEAN frame around c, corners drawn at their sub-pixel positions, placed away from the cursor."""
        R, Z = int(self.pc["loupe_radius"]), int(self.pc["loupe_zoom"])
        ix, iy = int(round(c[0])), int(round(c[1]))
        pad = cv2.copyMakeBorder(self.frame, R, R, R, R, cv2.BORDER_CONSTANT, value=0)
        big = cv2.resize(pad[iy:iy + 2 * R + 1, ix:ix + 2 * R + 1], ((2 * R + 1) * Z,) * 2,
                         interpolation=cv2.INTER_NEAREST)
        S = big.shape[0]
        if S + 8 > min(img.shape[:2]):
            return
        to = lambda x, y: (int(round((x - ix + R + 0.5) * Z)), int(round((y - iy + R + 0.5) * Z)))
        for i, p in enumerate(self.corners):
            u, v = to(*p)
            if -Z <= u <= S + Z and -Z <= v <= S + Z:
                col = (0, 255, 255) if i == self.sel else (0, 255, 0)
                cv2.circle(big, (u, v), max(3, Z // 2), col, 1, cv2.LINE_AA)
        cu, cv_ = to(c[0], c[1])
        for a, b in (((0, cv_), (cu - Z, cv_)), ((cu + Z, cv_), (S, cv_)), ((cu, 0), (cu, cv_ - Z)), ((cu, cv_ + Z), (cu, S))):
            cv2.line(big, a, b, (255, 255, 255), 1)
        cv2.rectangle(big, (0, 0), (S - 1, S - 1), (200, 200, 200), 1)
        cv2.putText(big, f"({c[0]:.1f}, {c[1]:.1f}) x{Z}", (4, S - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x0 = 4 if c[0] * self.s > img.shape[1] / 2 else img.shape[1] - S - 4
        img[4:4 + S, x0:x0 + S] = big

    # ------------------------------------------------------------------------------------------------------ editing
    def _mode(self, m):
        self.mode.set(m)
        self.redraw()

    def _motion(self, e):
        self.cursor = (e.x / self.s, e.y / self.s)
        self.loupe_on_sel = False
        if self.mode.get() == "corners":
            self.redraw()

    def _click(self, e):
        if self.preview_geom is not None or self.frame is None:
            return
        p = [e.x / self.s, e.y / self.s]
        if self.mode.get() == "corners":
            if len(self.corners) < 4:
                self.corners.append(p)
                self.sel = len(self.corners) - 1
            else:
                self.sel = int(np.argmin([np.hypot(c[0] - p[0], c[1] - p[1]) for c in self.corners]))
                self.corners[self.sel] = p
        else:
            self.lines[-1].append(p)
        self.redraw()

    def _tab(self, e):
        if self.corners:
            self.sel = (self.sel + 1) % len(self.corners)
            self.loupe_on_sel = True
            self.redraw()
        return "break"            # keep Tab from moving the keyboard focus

    def nudge(self, dx, dy):
        if self.mode.get() == "corners" and 0 <= self.sel < len(self.corners) and self.preview_geom is None:
            st = float(self.pc["nudge_px"])
            self.corners[self.sel][0] += dx * st
            self.corners[self.sel][1] += dy * st
            self.loupe_on_sel = True
            self.redraw()

    def new_line(self):
        if self.lines[-1]:
            self.lines.append([])
        self._mode("lines")

    def undo(self):
        if self.mode.get() == "corners" and self.corners:
            self.corners.pop()
            self.sel = len(self.corners) - 1
        elif self.mode.get() == "lines":
            if self.lines[-1]:
                self.lines[-1].pop()
            elif len(self.lines) > 1:
                self.lines.pop()
        self.redraw()

    def clear(self):
        self.corners, self.lines, self.sel = [], [[]], -1
        self.redraw()

    # ---------------------------------------------------------------------------------------------- auto-detection
    def auto(self):
        n = int(self.pc["detect_avg_frames"])
        self.status.configure(text=f"averaging {n} frames ...")
        acc, seen = [], set()

        def collect():
            pkt = self.engine.latest()
            if pkt is not None and pkt.index not in seen:
                seen.add(pkt.index)
                acc.append(pkt.raw.astype(np.float32))
            if len(acc) < n:
                self.after(10, collect)
                return
            img = np.clip(sum(acc) / len(acc), 0, 255).astype(np.uint8)   # noise falls as 1/sqrt(n)
            det = detect_target(img, self.pc["detect_contrast"], float(self.pc["detect_trim"]))
            if det is None:
                self.status.configure(text="auto-detect failed (see log); corners unchanged")
                return
            quad, edges, rms = det
            quad = order_corners(quad, self.corners or None, self.pc["first_corner"])
            if self.corners:
                log.info("auto-detect: shift vs previous corners %s px",
                         ", ".join(f"{np.hypot(*(quad[i] - np.array(self.corners[i]))):.1f}" for i in range(len(self.corners))))
            self.corners = [[round(float(x), 2), round(float(y), 2)] for x, y in quad]
            self.frame = img
            self.base = cv2.resize(img, self.disp, interpolation=cv2.INTER_LINEAR)
            self.sel, self.loupe_on_sel = 0, True
            self.mode.set("corners")
            log.info("auto-detect: corners %s; edge-fit rms %s px", "  ".join(f"({x:.1f},{y:.1f})" for x, y in self.corners),
                     ", ".join(f"{r:.2f}" for r in rms))
            self.redraw()
        collect()

    # ------------------------------------------------------------------------------------------------------ preview
    def preview(self):
        if len(self.corners) != 4:
            self.status.configure(text="need 4 corners")
            return
        rx = self.cfg["rectify"]
        try:
            g = Geometry.build(self.corners, [L for L in self.lines if L], (self.frame.shape[1], self.frame.shape[0]),
                               rx["out_size"], rx["margin"], rx["target_mm"], rx["radial"])
        except ValueError as e:
            self.status.configure(text=str(e))
            return
        self.preview_geom = g
        img = g.rectify(self.frame)
        from ..render import draw_grid
        img = draw_grid(img, g.mm_per_px, float(self.cfg["display"]["grid_mm"]))
        self.photo = to_photo(img, fit(img.shape[1], img.shape[0], *self.disp))
        self.canvas.configure(image=self.photo)
        self.pick_panel.pack_forget()
        self.prev_panel.pack(fill="x", before=self.status)
        k1 = "" if g.dist is None else f", k1 = {g.dist[0]:.4f}"
        self.status.configure(text=f"Rectified preview: {g.out_size} px, {g.mm_per_px:.4f} mm/px{k1}.\n"
                                   "Corner 1 is now top-left; the crosshair is the image centre.")

    def back(self):
        self.preview_geom = None
        self.prev_panel.pack_forget()
        self.pick_panel.pack(fill="x", before=self.status)
        self.redraw()

    def accept(self):
        if self.preview_geom is None:
            return self.preview()
        g = self.preview_geom
        self.destroy()
        self.on_done(g)

    def cancel(self):
        self.destroy()
        self.on_cancel()
