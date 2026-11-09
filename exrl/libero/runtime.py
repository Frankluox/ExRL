"""Locate an installed LIBERO package without changing Python's import path."""

from pathlib import Path
import importlib.util
import os
import yaml


def configure_runtime(root=None, config_dir=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise ImportError("Install LIBERO in this environment first; see docs/libero.md")
    package = Path(next(iter(spec.submodule_search_locations)))
    benchmark_root = package / "libero"
    if not (benchmark_root / "bddl_files").is_dir():
        raise FileNotFoundError(benchmark_root / "bddl_files")
    directory = Path(config_dir or os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    target = directory / "config.yaml"
    if not target.exists():
        target.write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(benchmark_root),
                    "bddl_files": str(benchmark_root / "bddl_files"),
                    "init_states": str(benchmark_root / "init_files"),
                    "assets": str(benchmark_root / "assets"),
                    "datasets": str(directory / "datasets"),
                }
            )
        )
    return package.parent
