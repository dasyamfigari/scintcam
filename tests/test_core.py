import json
import os
import time

import cv2
import numpy as np
import pytest

from conftest import FakeCap, plate_scene
from scintcam import analysis, render
from scintcam.config import ConfigError, load_config
from scintcam.detect import detect_target, order_corners
from scintcam.engine import CaptureEngine
from scintcam.geometry import Geometry, load_points, select_channel

CORNERS = ((150, 90), (470, 110), (450, 400), (130, 380))
HERE = os.path.dirname(__file__)


# ------------------------------------------------------------------------------------------------------- config
def test_example_config_loads():
    c = load_config(os.path.join(HERE, "..", "examples", "tandar_chromox.toml"))
    assert c["rectify"]["target_mm"] == 32.5 and c["rectify"]["channel"] == "r"
    assert c["camera"]["controls"]["sharpness"] == 0           # default kept
    assert c["camera"]["controls"]["power_line_frequency"] == 1


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[rectify]\ntarget_m = 30\n")
    with pytest.raises(ConfigError, match="rectify.target_m"):
        load_config(str(p))


# ----------------------------------------------------------------------------------------------------- geometry
def test_single_remap_matches_warp_perspective():
    frame = plate_scene()
    g = Geometry.build(CORNERS, [], (640, 480), 500, 0, 32.5, radial=False)
    ref = cv2.warpPerspective(frame, g.H, (500, 500))
    diff = np.abs(g.rectify(frame).astype(int) - ref.astype(int))
    assert np.percentile(diff, 99.9) <= 2           # fixed-point maps: at most a level or two at the plate edge
    assert g.mm_per_px == pytest.approx(32.5 / 500)


def test_rectified_plate_fills_output():
    g = Geometry.build(CORNERS, [], (640, 480), 400, 0, 32.5, radial=False)
    r = select_channel(g.rectify(plate_scene()), "r")
    assert r[5:-5, 5:-5].min() >= 160                # red plate everywhere inside


# ---------------------------------------------------------------------------------------------------- detection
def test_detect_corners_subpixel():
    quad, edges, rms = detect_target(plate_scene(), "r-g")
    q = order_corners(quad, None, "TL")
    assert np.max(np.linalg.norm(q - np.array(CORNERS, float), axis=1)) < 1.0
    assert max(rms) < 1.0


def test_detect_keeps_user_order():
    quad, _, _ = detect_target(plate_scene(), "r-g")
    mine = [CORNERS[2], CORNERS[3]]                       # user starts at bottom-right, then bottom-left
    q = order_corners(quad, mine)
    assert np.linalg.norm(q[0] - CORNERS[2]) < 1.5 and np.linalg.norm(q[1] - CORNERS[3]) < 1.5


def test_detect_rejects_round_beam_spot():
    img = np.full((480, 640, 3), 120, np.uint8)
    cv2.circle(img, (320, 240), 60, (40, 40, 230), -1)
    assert detect_target(cv2.GaussianBlur(img, (9, 9), 0), "r-g") is None


# ----------------------------------------------------------------------------------------------------- analysis
def test_fwhm_of_gaussian():
    sig = 20.0
    yy, xx = np.mgrid[0:300, 0:300]
    img = (200 * np.exp(-((xx - 150) ** 2 + (yy - 140) ** 2) / (2 * sig ** 2))).astype(np.float32)
    r = analysis.beam_readout(img, 0.1, 255)
    assert r["fwhm_x"] == pytest.approx(2.3548 * sig * 0.1, rel=0.02)
    assert r["ybar"] == pytest.approx(-1.0, abs=0.11)


def test_seismograph_keeps_short_dip():
    t = np.arange(0, 180, 0.04)
    I = np.full_like(t, 5e6)
    I[(t > 80) & (t < 80.08)] = 1e6                     # 2-frame beam trip among 4500 samples
    img = render.seismograph(t, I, np.zeros(t.size, int), t[-1], 3.0, 600, 170, [])
    green = (img[:, :, 1] > 150) & (img[:, :, 0] < 120)
    assert green[140:150].any()                         # the dip reaches the bottom of the plot


# ------------------------------------------------------------------------------------- engine + recorder outputs
def test_recording_outputs(cfg):
    cap = FakeCap(plate_scene(beam=60), period=1 / 60)
    eng = CaptureEngine(cap, cfg, (640, 480), 60.0)
    eng.start()
    g = Geometry.build(CORNERS, [], (640, 480), 300, 0, 32.5, radial=False)
    eng.set_geometry(g)
    time.sleep(0.8)                                     # let the measured rate settle (>= 30 frames)
    eng.record(True)
    time.sleep(1.0)
    eng.record(False)
    time.sleep(0.3)
    eng.stop()
    eng.join(5)
    last = eng.recorder.last
    assert last is not None and last["frames"] > 5         # loose: the machine running the tests may be busy
    d = cfg["recording"]["dir"]
    stem = last["stem"]
    names = set(os.listdir(d))
    for f in (f"{stem}.mkv", f"{stem}_rect.mkv", f"{stem}_frames.csv", f"{stem}_settings.json", f"{stem}.done"):
        assert f in names, f
    assert not [n for n in names if n.endswith(".tmp")]
    n_ts = len(open(os.path.join(d, f"{stem}_frames.csv")).read().splitlines()) - 1
    for v in (f"{stem}.mkv", f"{stem}_rect.mkv"):
        c = cv2.VideoCapture(os.path.join(d, v))
        k = 0
        while c.read()[0]:
            k += 1
        assert k == last["frames"] == n_ts, v       # every recorded frame has a timestamp
    done = json.load(open(os.path.join(d, f"{stem}.done")))
    assert done["files"] == [f"{stem}.mkv", f"{stem}_rect.mkv"]   # videos only (watchers pick the clip from here)
    st = json.load(open(os.path.join(d, f"{stem}_settings.json")))
    assert st["status"] == "complete" and st["rectification"]["mm_per_px"] == pytest.approx(32.5 / 300)
    assert os.path.getmtime(os.path.join(d, f"{stem}.done")) >= os.path.getmtime(os.path.join(d, f"{stem}_settings.json"))
    corners, _ = load_points(cfg["rectify"]["points_json"], f"{stem}.mkv")
    assert corners is not None                         # raw clip registered for offline tools
    t, I, s, marks = eng.seismo.data()
    assert I.size > 50 and [m[1] for m in marks] == ["REC", "STOP"]


def test_gui_imports():
    pytest.importorskip("tkinter")
    pytest.importorskip("PIL.ImageTk")
    from scintcam.gui import app, picker  # noqa: F401


def test_save_controls_edits_only_camera_controls(tmp_path):
    from scintcam.config import save_controls
    p = tmp_path / "c.toml"
    p.write_text("[camera.controls]\ngain = 4        # keep me\n\n[rectify]\ntarget_mm = 32.5\n")
    save_controls(str(p), {"gain": 10, "exposure": 125, "auto_exposure": False})
    save_controls(str(p), {"exposure": 200})
    txt = p.read_text()
    cfg = load_config(str(p))
    assert cfg["camera"]["controls"]["gain"] == 10 and cfg["camera"]["controls"]["exposure"] == 200
    assert cfg["camera"]["controls"]["auto_exposure"] is False and cfg["rectify"]["target_mm"] == 32.5
    assert "# keep me" in txt and txt.count("\nexposure =") == 1
