"""Capture exactly what produced a result: code version, config, environment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def git_hash(short: bool = False) -> str:
    """Current commit hash, with ``-dirty`` appended if the tree has changes."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        h = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"{h}-dirty" if dirty else h
    except Exception:  # pragma: no cover - git may be absent
        return "unknown"


def environment() -> dict[str, Any]:
    import numpy as np
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def capture_provenance(
    out_path: str | os.PathLike[str],
    config: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<out_path>`` with git hash, environment, config and argv."""
    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_hash": git_hash(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "environment": environment(),
    }
    if config is not None:
        record["config"] = config.to_dict() if hasattr(config, "to_dict") else config
    if extra:
        record["extra"] = extra
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record
