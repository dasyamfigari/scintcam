"""Command line:  scintcam [--config FILE]  |  scintcam probe|diagnose [--config FILE]  |  scintcam --print-config"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

from . import __version__
from .config import ConfigError, default_text, load_config


def _logging(log_dir: str = "") -> None:
    lg = logging.getLogger("scintcam")
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    lg.addHandler(h)
    if log_dir:   # session log next to the recordings
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, datetime.datetime.now().strftime("scintcam_%Y-%m-%d.log")))
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        lg.addHandler(fh)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="scintcam", description="Live rectified view and recording of a square target "
                                                             "(e.g. a scintillator screen) seen by a webcam.")
    p.add_argument("command", nargs="?", default="gui", choices=["gui", "probe", "diagnose"],
                   help="gui (default); probe: list the modes the camera really streams; diagnose: test what each "
                        "camera control really does and write the camera profile")
    p.add_argument("-o", "--output", default="", help="diagnose: profile file (default: camera.profile)")
    p.add_argument("-c", "--config", help="TOML file with the settings that differ from the defaults")
    p.add_argument("--print-config", action="store_true", help="print the full default configuration and exit")
    p.add_argument("--version", action="version", version=f"scintcam {__version__}")
    a = p.parse_args(argv)
    if a.print_config:
        print(default_text())
        return 0
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if a.command == "probe":
        from .camera import probe
        print(probe(cfg["camera"]))
        return 0
    if a.command == "diagnose":
        from .diagnose import run as diagnose
        return diagnose(cfg, a.output)
    if not cfg["rectify"]["target_mm"] or cfg["rectify"]["target_mm"] <= 0:
        print("config error: set rectify.target_mm (side of the square target in mm) in your config file",
              file=sys.stderr)
        return 2
    _logging(cfg["recording"]["dir"])
    from .gui.app import run_gui
    run_gui(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
