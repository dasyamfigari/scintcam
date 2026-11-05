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


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(v)


def save_controls(path: str, values: Dict[str, Any]) -> None:
    """Write `values` into the [camera.controls] table of the TOML file at `path`, editing only those lines.

    A key already set in the table keeps its line (and its trailing comment) with the new value; a missing key is
    added at the end of the table; a missing table is appended. Commented-out lines are left alone. The result is
    parsed before it replaces the file, so a failure never leaves a broken config.
    """
    import re
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    head = re.compile(r"^\s*\[\s*camera\s*\.\s*controls\s*\]\s*(#.*)?$")
    table = re.compile(r"^\s*\[")
    start = next((i for i, l in enumerate(lines) if head.match(l.rstrip("\r\n"))), None)
    todo = dict(values)
    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["\n", "[camera.controls]\n"] + [f"{k} = {_toml_value(v)}\n" for k, v in todo.items()]
    else:
        end = next((i for i in range(start + 1, len(lines)) if table.match(lines[i])), len(lines))
        for i in range(start + 1, end):
            m = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)([^#\r\n]*?)(\s*#.*)?(\r?\n)?$", lines[i])
            if m and m.group(2) in todo:
                lines[i] = (m.group(1) + m.group(2) + m.group(3) + _toml_value(todo.pop(m.group(2)))
                            + (("  " + m.group(5).strip()) if m.group(5) else "") + (m.group(6) or "\n"))
        last = end
        while last > start + 1 and not lines[last - 1].strip():   # insert before the blank lines that end the table
            last -= 1
        lines[last:last] = [f"{k} = {_toml_value(v)}\n" for k, v in todo.items()]
    text = "".join(lines)
    tomllib.loads(text)                     # refuse to write anything that does not parse
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
