"""Locations of the source tree and bundled dependencies."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "dsrl"
