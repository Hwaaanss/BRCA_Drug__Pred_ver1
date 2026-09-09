"""Structured logging.

Human-readable lines go to stderr; machine-readable records go to a JSON-lines
file so that every number in a report can be traced back to a run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

_CONFIGURED = False
_LOCK = threading.Lock()


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("hill")
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"hill.{name}")


class JsonlLogger:
    """Append-only JSON-lines sink.

    Records are flushed immediately so a killed run still leaves a usable trace.
    """

    def __init__(self, path: str | os.PathLike[str], run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "run_id": self.run_id, "event": event}
        record.update(fields)
        line = json.dumps(record, default=_json_default, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
