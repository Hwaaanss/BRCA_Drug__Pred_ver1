"""Runtime / resource management for a single-GPU node.

Target machine for this project:
    GPU   1x NVIDIA A100 80GB, index 0 only
    RAM   100 GB
    DISK  30 GB
    CPU   10 cores

The helpers here pin the process to GPU 0, cap thread counts so the 10-core
budget is not oversubscribed, cap the CUDA allocator, and refuse to start a
long run when the disk is already nearly full.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from hill.utils.logging import get_logger

log = get_logger("resources")

_GB = 1024**3


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GB
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
        return float("nan")


def available_ram_gb() -> float:
    """MemAvailable from /proc, falling back to total RAM."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024**2)
    except OSError:  # pragma: no cover - non-Linux
        pass
    return total_ram_gb()


def free_disk_gb(path: str | os.PathLike[str] = ".") -> float:
    try:
        return shutil.disk_usage(str(path)).free / _GB
    except OSError:  # pragma: no cover
        return float("nan")


def configure_runtime(cfg: Any, require_gpu: bool = False) -> dict[str, Any]:
    """Apply the resource policy in ``cfg.resources`` and return a description.

    Must be called before any CUDA tensor is created for the device pinning to
    take effect.
    """
    import torch

    res = cfg.resources

    # --- pin to a single GPU -------------------------------------------------
    if not torch.cuda.is_initialized():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(res.gpu_index))
    elif os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(res.gpu_index)):
        log.warning(
            "CUDA already initialised; CUDA_VISIBLE_DEVICES=%s left as-is",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )

    # --- CPU thread budget ---------------------------------------------------
    threads = max(1, min(res.torch_threads, os.cpu_count() or res.torch_threads))
    torch.set_num_threads(threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))

    # --- GPU allocator -------------------------------------------------------
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.cuda.set_per_process_memory_fraction(res.max_gpu_memory_fraction, 0)
        except (AssertionError, RuntimeError) as exc:  # pragma: no cover
            log.warning("could not cap GPU memory fraction: %s", exc)
    elif require_gpu:
        raise RuntimeError(
            "CUDA is required for this command but torch.cuda.is_available() is False. "
            "Check the driver / CUDA build (see README troubleshooting)."
        )
    else:
        log.warning("running on CPU — this is only supported for tests and the smoke test")

    # --- host resource sanity ------------------------------------------------
    ram_avail = available_ram_gb()
    if ram_avail < res.max_ram_gb * (1.0 - res.ram_warn_fraction):
        log.warning("only %.1f GB RAM available (budget %.0f GB)", ram_avail, res.max_ram_gb)
    disk = free_disk_gb(cfg.paths.results_dir if Path(cfg.paths.results_dir).exists() else ".")
    if disk < 2.0:
        raise RuntimeError(
            f"only {disk:.1f} GB free disk; refusing to start (budget {res.max_disk_gb:.0f} GB). "
            "Delete old runs under results/ first."
        )

    desc = describe_runtime(cfg)
    log.info(
        "runtime: device=%s threads=%d ram_avail=%.0fGB disk_free=%.0fGB",
        desc["device"], threads, ram_avail, disk,
    )
    return desc


def describe_runtime(cfg: Any = None) -> dict[str, Any]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg is not None and getattr(cfg, "device", "cuda") == "cpu":
        device = "cpu"
    out: dict[str, Any] = {
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "ram_total_gb": round(total_ram_gb(), 1),
        "ram_available_gb": round(available_ram_gb(), 1),
        "disk_free_gb": round(free_disk_gb(), 1),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out["gpu_name"] = props.name
        out["gpu_total_gb"] = round(props.total_memory / _GB, 1)
    return out


def resolve_device(cfg: Any) -> "Any":
    import torch

    if cfg.device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def amp_dtype(cfg: Any) -> "Any | None":
    import torch

    kind = cfg.train.amp_dtype
    if kind == "none" or not torch.cuda.is_available():
        return None
    if kind == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log.warning("bf16 unsupported on this GPU; falling back to fp16")
        return torch.float16
    return torch.float16


def dataloader_workers(cfg: Any) -> int:
    """Leave one core for the main process and any GPU feeder threads."""
    budget = max(0, min(cfg.train.num_workers, (os.cpu_count() or 2) - 1))
    return budget
