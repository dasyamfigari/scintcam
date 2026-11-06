"""scintcam diagnose against simulated UVC cameras (tests/fakeuvc.py) that behave in the ways real ones do."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import fakeuvc  # noqa: E402

from scintcam import camera as C  # noqa: E402
from scintcam import diagnose as D  # noqa: E402
from scintcam.config import load_config  # noqa: E402


@pytest.fixture
def fakecam(tmp_path, monkeypatch):
    if sys.platform.startswith("win"):
        pytest.skip("simulated v4l2-ctl is a shell wrapper")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    w = bindir / "v4l2-ctl"
    w.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{os.path.join(os.path.dirname(__file__), "fakeuvc.py")}" "$@"\n')
    w.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(C, "IS_LINUX", True)
    state = tmp_path / "state.json"
    monkeypatch.setenv("FAKE_UVC", str(state))
    monkeypatch.setattr(D, "SCALE", 0.1)
    monkeypatch.setattr(C, "open_camera", lambda cam: (fakeuvc.FakeUVCCap(D.SCALE), (160, 120), 30.0))
    return state


DV20 = dict(unit_s=1.31e-6, k=0.00657, exp_range=(80, 100000))   # the TANDAR webcam, as its diagnosis measured it


@pytest.mark.parametrize("behaviour,readback,clamp,extra,apply,auto_rb,after", [
    ("latch_stored", "stale", False, {}, "store_then_cycle", "stale", "returns_to_stored"),
    ("keeps_auto", "live", True, {}, "direct", "live", "keeps_auto"),
    ("keeps_auto", "stale", True, DV20, "direct", "stale", "keeps_auto"),
    ("keeps_auto", "stale", True, dict(DV20, ignore_same=True), "direct", "stale", "keeps_auto"),
])
def test_diagnose_classifies_camera(fakecam, tmp_path, behaviour, readback, clamp, extra, apply, auto_rb, after):
    fakeuvc.make_state(str(fakecam), behaviour, readback, clamp, **extra)
    out = tmp_path / "camera_profile.json"
    assert D.run(load_config(None), str(out)) == 0
    p = json.loads(out.read_text())
    ex = p["exposure"]
    assert ex["apply"] == apply and ex["auto_readback"] == auto_rb and ex["manual_after_auto"] == after
    assert ex["out_of_range"] == ("clamped" if clamp else "accepted")
    assert ex["lock_from_auto"] == "ok"                     # start-up lock from auto, target already stored
    assert ex["response"]["points"] >= 3 and ex["response"]["ms_min"] < ex["response"]["ms_max"]
    unit_true = 1e3 * extra.get("unit_s", 1e-4)              # exposure unit measured from the frame rate
    assert abs(ex["unit_ms_measured"] / unit_true - 1) < 0.1
    assert 28.0 <= ex["full_rate_max_ms"] <= 40.0          # sensor slows down above 1/30 s (sim timing jitter)
    assert ex["response"]["r2"] > 0.99
    eff = {n: e.get("effect") for n, e in p["controls"].items()}
    assert eff["gain"] == "level" and eff["saturation"] == "colour" and eff["sharpness"] == "none"
    assert (tmp_path / "camera_profile.txt").exists()
    st = json.loads(fakecam.read_text())                     # every control restored (a camera that starts in
    skip = {"exposure_absolute"} if readback == "live" else set()   # auto and reports live gets its auto value back)
    assert {k: d["value"] for k, d in st["controls"].items() if k not in skip} == \
        {k: d["value"] for k, d in fakeuvc.CONTROLS.items() if k not in skip}


def test_profile_sets_exposure_strategy(fakecam, tmp_path):
    fakeuvc.make_state(str(fakecam), "latch_stored", "stale", True)
    cfg = load_config(None)
    cfg["camera"]["profile"] = str(tmp_path / "p.json")
    (tmp_path / "p.json").write_text(json.dumps({"device": {"card": "Fake UVC Camera: Fake"},
                                                 "exposure": {"apply": "store_then_cycle"}}))
    C.load_profile(cfg)
    assert C.exposure_strategy(cfg) == "store_then_cycle"
    cfg["camera"]["controls"].update(auto_exposure=False, exposure=420)
    C.lock_camera(None, cfg)
    assert json.loads(fakecam.read_text())["active"] == 420  # in effect on the first application
    snap = C.snapshot(cfg)                                    # ladder from the declared minimum (3) x 3
    assert snap["summary"]["exposure_ladder"][:3] == [3, 9, 27] and snap["summary"]["exposure_rung"] is None
    (tmp_path / "p.json").write_text(json.dumps({"device": {"card": "Another camera"}, "exposure": {"apply": "direct"}}))
    assert C.load_profile(cfg) is None and C.exposure_strategy(cfg) == "full_sequence"


def test_lock_from_auto_with_target_already_stored(fakecam, tmp_path):
    """DV20-like camera: auto -> manual keeps the auto exposure without updating the stored number, and a write equal
    to the stored number is skipped. The lock must still put the target into effect (neighbour write first)."""
    fakeuvc.make_state(str(fakecam), "keeps_auto", "stale", True, ignore_same=True, **DV20)
    st = json.loads(fakecam.read_text())
    st["controls"]["exposure_absolute"]["value"] = 30000     # target already stored, camera in auto
    fakecam.write_text(json.dumps(st))
    cfg = load_config(None)
    cfg["camera"]["profile"] = str(tmp_path / "p.json")
    (tmp_path / "p.json").write_text(json.dumps({"device": {"card": "Fake UVC Camera: Fake"},
                                                 "exposure": {"apply": "direct"}}))
    C.load_profile(cfg)
    cfg["camera"]["controls"].update(auto_exposure=False, exposure=30000)
    C.lock_camera(None, cfg)
    st = json.loads(fakecam.read_text())
    assert st["controls"]["exposure_auto"]["value"] == 1 and st["active"] == 30000
