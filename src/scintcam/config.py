"""Configuration: packaged defaults (default_config.toml) deep-merged with an optional user TOML file."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # novm  (Python 3.11+; tomli below otherwise)
except ModuleNotFoundError:  # pragma: no cover - Python 3.8-3.10
    import tomli as tomllib

DEFAULT_PATH = Path(__file__).with_name("default_config.toml")

# Tables whose keys are free (not checked against the defaults).
FREE_TABLES = {("camera", "controls")}

KEEP = "keep"  # control value meaning "do not touch"


class ConfigError(ValueError):
    pass


def default_text() -> str:
    return DEFAULT_PATH.read_text(encoding="utf-8")


def _merge(base: Dict[str, Any], over: Dict[str, Any], path: tuple = ()) -> None:
    """Merge `over` into `base` in place; unknown keys are an error (typos must not pass silently)."""
    for key, val in over.items():
        here = path + (key,)
        if path in FREE_TABLES:
            base[key] = val
            continue
        if key not in base:
            raise ConfigError(f"unknown setting '{'.'.join(here)}' (see scintcam --print-config)")
        if isinstance(base[key], dict):
            if not isinstance(val, dict):
                raise ConfigError(f"'{'.'.join(here)}' must be a table")
            _merge(base[key], val, here)
        else:
            base[key] = val


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = tomllib.loads(default_text())
    if path:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "rb") as f:
            user = tomllib.load(f)
        _merge(cfg, user)
        cfg["_path"] = str(p.resolve())
    else:
        cfg["_path"] = None
    return cfg


def controls(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Camera controls to write (the ones set to "keep" are dropped)."""
    return {k: v for k, v in cfg["camera"]["controls"].items() if v != KEEP}


def snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
