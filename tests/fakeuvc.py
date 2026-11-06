"""Simulated UVC camera for tests: a fake `v4l2-ctl` (run as a script) and a fake capture sharing one state file.

The image level follows the exposure ACTIVE in the simulated sensor, which differs from the stored number in the
ways real cameras differ:
  behaviour = "direct"        manual writes apply at once; auto -> manual uses the stored value
              "latch_stored"  manual writes are only stored; auto -> manual latches the stored value
              "keeps_auto"    manual writes apply at once; auto -> manual keeps the auto exposure (and stores it)
  readback  = "live" | "stale"  what exposure_absolute reports while auto is on
  clamp     = True | False      out-of-range writes clamped (uvcvideo) or accepted as is
  unit_s    = duration of one exposure step [s] (UVC standard 1e-4; the TANDAR DV20 webcam about 1.3e-6)
  k         = level per exposure step [counts]
A camera with stale read-back that keeps the auto exposure does not update the stored register either (DV20).
"""
import json
import os
import sys
import time

import numpy as np

CONTROLS = {
    "brightness": dict(type="int", min=-64, max=64, step=1, default=0, value=0),
    "contrast": dict(type="int", min=0, max=64, step=1, default=32, value=32),
    "saturation": dict(type="int", min=0, max=128, step=1, default=64, value=64),
    "gamma": dict(type="int", min=72, max=500, step=1, default=100, value=100),
    "gain": dict(type="int", min=0, max=100, step=1, default=0, value=0),
    "sharpness": dict(type="int", min=0, max=6, step=1, default=3, value=3),
    "white_balance_temperature_auto": dict(type="bool", default=1, value=1),
    "white_balance_temperature": dict(type="int", min=2800, max=6500, step=1, default=4600, value=4600),
    "power_line_frequency": dict(type="menu", min=0, max=2, default=1, value=1),
    "exposure_auto": dict(type="menu", min=0, max=3, default=3, value=3),
    "exposure_absolute": dict(type="int", min=3, max=2047, step=1, default=250, value=250),
    "exposure_auto_priority": dict(type="bool", default=0, value=0),
}


def make_state(path, behaviour="direct", readback="live", clamp=True, unit_s=1e-4, k=0.4, exp_range=None):
    ctl = json.loads(json.dumps(CONTROLS))
    if exp_range:
        ctl["exposure_absolute"].update(min=exp_range[0], max=exp_range[1])
    st = {"controls": ctl, "behaviour": behaviour, "readback": readback, "clamp": clamp, "unit_s": unit_s, "k": k,
          "active": ctl["exposure_absolute"]["value"]}
    with open(path, "w") as f:
        json.dump(st, f)
    return path


def _load():
    with open(os.environ["FAKE_UVC"]) as f:
        return json.load(f)


def _save(st):
    tmp = os.environ["FAKE_UVC"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, os.environ["FAKE_UVC"])


def _v(st, k):
    return st["controls"][k]["value"]


def level_of(st, active):
    """Mean grey level for an active exposure (units of 100 us) and the current gain/brightness/contrast/gamma."""
    lv = 8 + 0.5 * _v(st, "brightness") + st.get("k", 0.4) * active * (1 + _v(st, "gain") / 25.0) * _v(st, "contrast") / 32.0
    lv = min(max(lv, 0.0), 255.0)
    return 255.0 * (lv / 255.0) ** (100.0 / _v(st, "gamma"))


def auto_target(st):
    """Exposure the auto loop settles on: level 128 (solved on the linear part, before gamma)."""
    k = st.get("k", 0.4) * (1 + _v(st, "gain") / 25.0) * _v(st, "contrast") / 32.0
    d = st["controls"]["exposure_absolute"]
    return int(min(max((128 - 8 - 0.5 * _v(st, "brightness")) / max(k, 1e-9), d["min"]), d["max"]))


def in_use(st):
    return auto_target(st) if _v(st, "exposure_auto") == 3 else st["active"]


def ctl_main(argv):
    a = list(argv)
    if "-d" in a:
        i = a.index("-d")
        del a[i:i + 2]
    st = _load()
    c = st["controls"]
    if a and a[0] in ("-l", "-L"):
        for k, d in c.items():
            fl = ""
            if k == "exposure_absolute" and _v(st, "exposure_auto") == 3:
                fl = " flags=inactive"
            if k == "white_balance_temperature" and _v(st, "white_balance_temperature_auto") == 1:
                fl = " flags=inactive"
            val = d["value"]
            if k == "exposure_absolute" and _v(st, "exposure_auto") == 3 and st["readback"] == "live":
                val = auto_target(st)
            rng = f"min={d['min']} max={d['max']} step={d.get('step', 1)} " if "min" in d and d["type"] != "bool" else ""
            print(f"{k:>32} 0x00980900 ({d['type']}){'':4}: {rng}default={d['default']} value={val}{fl}")
        return 0
    if a and a[0] == "-C":
        k = a[1]
        if k not in c:
            print(f"unknown control '{k}'", file=sys.stderr)
            return 1
        val = c[k]["value"]
        if k == "exposure_absolute" and _v(st, "exposure_auto") == 3 and st["readback"] == "live":
            val = auto_target(st)
        print(f"{k}: {val}")
        return 0
    if a and a[0] == "-c":
        k, v = a[1].split("=")
        v = int(v)
        if k not in c:
            print(f"unknown control '{k}'", file=sys.stderr)
            return 1
        d = c[k]
        if "min" in d and st["clamp"]:
            v = min(max(v, d["min"]), d["max"])
        beh = st["behaviour"]
        if k == "exposure_auto":
            old = d["value"]
            d["value"] = v
            if old == 3 and v == 1:
                if beh == "keeps_auto":
                    st["active"] = auto_target(st)
                    if st["readback"] == "live":
                        c["exposure_absolute"]["value"] = st["active"]
                else:
                    st["active"] = c["exposure_absolute"]["value"]
        elif k == "exposure_absolute":
            d["value"] = v
            if _v(st, "exposure_auto") == 1 and beh in ("direct", "keeps_auto"):
                st["active"] = v
        else:
            d["value"] = v
        _save(st)
        return 0
    if a and a[0] == "--info":
        print("Driver Info:\n\tDriver name      : uvcvideo\n\tCard type        : Fake UVC Camera: Fake\n"
              "\tBus info         : usb-0000:00:14.0-1\n\tDriver version   : 5.4.0")
        return 0
    if a and a[0] == "--get-parm":
        print("Streaming Parameters Video Capture:\n\tFrames per second: 30.000 (30/1)")
        return 0
    if a and a[0] == "--get-fmt-video":
        print("Format Video Capture:\n\tWidth/Height      : 160/120\n\tPixel Format      : 'YUYV' (YUYV 4:2:2)")
        return 0
    print("")
    return 0


class FakeUVCCap:
    """cv2.VideoCapture stand-in whose frames follow the simulated sensor; frame period follows the exposure."""

    def __init__(self, scale=1.0):
        self.scale = scale
        rng = np.random.default_rng(1)
        self.noise = [rng.normal(0, 0.8, (120, 160, 3)).astype(np.float32) for _ in range(4)]
        self.n = 0

    def read(self):
        st = _load()
        a = in_use(st)
        time.sleep(max(1 / 30.0, a * st.get("unit_s", 1e-4)) * self.scale)
        lv = level_of(st, a)
        sat = _v(st, "saturation") / 64.0
        wb = 0.0 if _v(st, "white_balance_temperature_auto") else (_v(st, "white_balance_temperature") - 4600) / 4600
        ratios = np.array([0.85 + wb * 0.3, 1.0, 1.15 - wb * 0.3])
        bgr = np.clip(lv * (1 + (ratios - 1) * sat), 0, 255)
        self.n += 1
        f = np.clip(bgr[None, None, :] + self.noise[self.n % 4], 0, 255).astype(np.uint8)
        return True, f

    def get(self, prop):
        return 0.0

    def set(self, prop, val):
        return True

    def isOpened(self):
        return True

    def release(self):
        pass


if __name__ == "__main__":
    sys.exit(ctl_main(sys.argv[1:]))
