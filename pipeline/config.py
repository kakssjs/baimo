"""Loads config.toml from the project root."""
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "jobs"

with open(ROOT / "config.toml", "rb") as f:
    _cfg = tomllib.load(f)

LIMITS = _cfg["limits"]
WHITEBOX_SCRIPTS = _cfg["paths"]["whitebox_scripts"]
DA3_SRC = str((ROOT / _cfg["paths"]["depth_anything_3"]).resolve())


def tool(name):
    """Resolve a [tools] entry to an executable path, or fail with a clear message."""
    value = _cfg["tools"][name]
    path = shutil.which(value)
    if not path:
        raise RuntimeError(f"{name} not found: '{value}' (set [tools].{name} in config.toml)")
    return path
