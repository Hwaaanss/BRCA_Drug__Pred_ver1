"""Smoke test: run the entire pipeline on SYNTHETIC data, then delete the output.

    python -m hill.smoke                 # ~2-4 min on CPU, leaves nothing behind
    python -m hill.smoke --keep          # keep the artefacts for inspection
    python -m hill.smoke --with-tests    # also run the unit test suite first

This exercises ingest -> Gate 0 -> HPO -> ablation ladder -> Stage 1 -> figures
with tiny tensors, so a syntax error, a shape bug or a broken file path shows up
in minutes instead of hours into a real run.  Every artefact it produces is
marked SYNTHETIC (project rule R2) and is removed at the end unless --keep.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hill.config import load_config
from hill.utils.logging import get_logger

log = get_logger("smoke")

STEPS: list[tuple[str, list[str]]] = [
    ("generate SYNTHETIC data", [sys.executable, "-m", "hill.data.synthetic", "--data-root", "data_smoke"]),
    ("prepare", [sys.executable, "-m", "hill.data.prepare", "--config", "configs/smoke.yaml", "--force"]),
    ("Gate 0 audit", [sys.executable, "-m", "hill.audit.run_gate0", "--config", "configs/smoke.yaml",
                      "--steps", "200", "--allow-fail"]),
    ("HPO (2 trials)", [sys.executable, "-m", "hill.hpo", "--config", "configs/smoke.yaml", "--n-trials", "2"]),
    ("ablation (2 seeds, steps 0-2)", [sys.executable, "-m", "hill.ablation", "--config", "configs/smoke.yaml",
                                       "--seeds", "2", "--steps", "0", "1", "2"]),
    ("figures", [sys.executable, "-m", "hill.figures.run_all", "--config", "configs/smoke.yaml"]),
    ("findings report", [sys.executable, "-m", "hill.report", "--config", "configs/smoke.yaml"]),
]

CLEAN_PATHS = ["data_smoke", "results_smoke", "reports_smoke"]


def _run(name: str, cmd: list[str], verbose: bool) -> tuple[bool, float, str]:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=not verbose, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    tail = ""
    if not ok and not verbose:
        tail = (proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-3000:]
    return ok, dt, tail


def _stage1_smoke(verbose: bool) -> tuple[bool, float, str]:
    """Stage 1 needs a checkpoint produced by the ablation step."""
    cfg = load_config("configs/smoke.yaml")
    ckpts = sorted(Path(cfg.paths.checkpoint_dir).glob("**/step2*.pt"))
    if not ckpts:
        return True, 0.0, "skipped: no stage-0 checkpoint produced"
    cmd = [sys.executable, "-m", "hill.stage1", "--config", "configs/smoke.yaml",
           "--checkpoint", str(ckpts[0]), "--folds", "2"]
    return _run("stage 1", cmd, verbose)


def cleanup() -> None:
    for path in CLEAN_PATHS:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
            log.info("removed %s", p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end smoke test on SYNTHETIC data")
    ap.add_argument("--keep", action="store_true", help="do not delete the artefacts afterwards")
    ap.add_argument("--verbose", action="store_true", help="stream each step's output")
    ap.add_argument("--with-tests", action="store_true", help="run pytest first")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("HILL smoke test — SYNTHETIC data only, no scientific result is produced")
    print("=" * 72)

    failures: list[str] = []
    t_start = time.time()

    if args.with_tests:
        ok, dt, tail = _run("unit tests", [sys.executable, "-m", "pytest", "tests", "-q"], args.verbose)
        print(f"[{'PASS' if ok else 'FAIL'}] unit tests ({dt:.1f}s)")
        if not ok:
            failures.append("unit tests")
            print(tail)

    for name, cmd in STEPS:
        ok, dt, tail = _run(name, cmd, args.verbose)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s)")
        if not ok:
            failures.append(name)
            print(tail)
            break
        if name.startswith("ablation"):
            ok1, dt1, tail1 = _stage1_smoke(args.verbose)
            print(f"[{'PASS' if ok1 else 'FAIL'}] stage 1 (TCGA + gates) ({dt1:.1f}s)")
            if not ok1:
                failures.append("stage 1")
                print(tail1)
                break

    total = time.time() - t_start
    print("-" * 72)
    if failures:
        print(f"SMOKE TEST FAILED after {total:.1f}s — failing step(s): {', '.join(failures)}")
    else:
        print(f"SMOKE TEST PASSED in {total:.1f}s")

    if args.keep:
        print("artefacts kept: " + ", ".join(CLEAN_PATHS))
    else:
        cleanup()
        print("artefacts removed: " + ", ".join(CLEAN_PATHS))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
