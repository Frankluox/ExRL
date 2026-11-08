from __future__ import annotations

import torch


def auto_select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    del log
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but unavailable: {try_device}")
        return torch.device(try_device)
    if try_device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but unavailable")
        return torch.device("mps")
    if try_device == "cpu":
        return torch.device("cpu")
    return torch.device(try_device)


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    if try_device == "mps":
        return torch.backends.mps.is_available()
    if try_device == "cpu":
        return True
    try:
        torch.device(try_device)
    except Exception:
        return False
    return True


def is_amp_available(device: str) -> bool:
    return str(device) != "mps"
