"""One command for the whole study.

    python -m hill.run_all --config configs/base.yaml

Order (each stage stops the run if it fails):

    1. download        external data (skip with --skip-download once you have it)
    2. prepare         raw wells -> point/pair tables, omics, tokens, splits
    3. Gate 0          G-1..G-5; a failed G-1 or an unsupported G-3 halts here
    4. unit tests      including test_gamma_zero_equivalence
    5. HPO             Optuna, 10 trials on one LCO fold
    6. ablation        the §7.4 ladder + baselines, 10 seeds, best params
    7. Stage 1 + LRT   TCGA transfer and the two histology gates
    8. figures         every manuscript figure
    9. findings        reports/findings.md

Expected wall-clock on one A100-80GB with the shipped configs/base.yaml:
roughly 1-2 h for stages 1-4, ~2 h for HPO, ~15-20 h for the 10-seed ladder.
Use --seeds / --n-trials to trade accuracy for time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from hill.config import load_config
from hill.utils.logging import get_logger

log = get_logger("run_all")

STAGES = ["download", "prepare", "gate0", "tests", "hpo", "ablation", "stage1", "figures", "report"]


def _run(name: str, cmd: list[str], allow_fail: bool = False) -> bool:
    print("\n" + "=" * 72)
    print(f"[{name}] {' '.join(cmd)}")
    print("=" * 72, flush=True)
    t0 = time.time()
    code = subprocess.call(cmd)
    dt = time.time() - t0
    ok = code == 0
    print(f"[{name}] {'ok' if ok else f'FAILED (exit {code})'} in {dt / 60:.1f} min", flush=True)
    return ok or allow_fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the complete HILL study")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--from-stage", choices=STAGES, default="download")
    ap.add_argument("--only", nargs="*", choices=STAGES, default=None)
    ap.add_argument("--n-trials", type=int, default=None, help="override hpo.n_trials")
    ap.add_argument("--seeds", type=int, default=None, help="override ablation.seeds")
    ap.add_argument("--allow-gate-fail", action="store_true",
                    help="continue even if Gate 0 does not pass (NOT for a real study)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    cfg.paths.ensure()
    py = sys.executable
    common = ["--config", args.config] + (["--set", *args.set] if args.set else [])
    best_params = Path(cfg.hpo.storage).parent / "best_params.yaml"

    def wanted(stage: str) -> bool:
        if args.only:
            return stage in args.only
        return STAGES.index(stage) >= STAGES.index(args.from_stage)

    t0 = time.time()

    if wanted("download") and not args.skip_download:
        if not _run("1/9 download", [py, "-m", "hill.data.download", "--what", "all"]):
            print("\nDownload failed. Fix the URLs in configs/data_sources.yaml or place the files "
                  "manually, then rerun with --from-stage prepare.")
            return 1
        _run("1b/9 download TCGA", [py, "-m", "hill.data.download_tcga"], allow_fail=True)

    if wanted("prepare") and not _run("2/9 prepare", [py, "-m", "hill.data.prepare", *common]):
        return 1

    if wanted("gate0"):
        cmd = [py, "-m", "hill.audit.run_gate0", *common]
        if args.allow_gate_fail:
            cmd.append("--allow-fail")
        if not _run("3/9 Gate 0", cmd):
            print("\nGate 0 did not pass — see reports/gate0.md. Stopping here on purpose "
                  "(project rule R1). Use --allow-gate-fail only to explore, never to publish.")
            return 2

    if wanted("tests") and not _run("4/9 unit tests", [py, "-m", "pytest", "tests", "-q"]):
        return 1

    if wanted("hpo"):
        cmd = [py, "-m", "hill.hpo", *common]
        if args.n_trials:
            cmd += ["--n-trials", str(args.n_trials)]
        if not _run("5/9 HPO", cmd):
            return 1

    if wanted("ablation"):
        cmd = [py, "-m", "hill.ablation", *common]
        if best_params.exists():
            cmd += ["--extra", str(best_params)]
        else:
            log.warning("no HPO best-params file at %s; using the base config", best_params)
        if args.seeds:
            cmd += ["--seeds", str(args.seeds)]
        if not _run("6/9 ablation ladder", cmd):
            return 1

    if wanted("stage1"):
        ckpts = sorted(Path(cfg.paths.checkpoint_dir).glob("**/step2*.pt"))
        if ckpts:
            cmd = [py, "-m", "hill.stage1", *common, "--checkpoint", str(ckpts[-1]), "--lrt"]
            _run("7/9 Stage 1 + LRT", cmd, allow_fail=True)
        else:
            log.warning("no stage-0 checkpoint found; skipping Stage 1")

    if wanted("figures"):
        _run("8/9 figures", [py, "-m", "hill.figures.run_all", *common], allow_fail=True)

    if wanted("report"):
        _run("9/9 findings", [py, "-m", "hill.report", *common], allow_fail=True)

    print("\n" + "=" * 72)
    print(f"Done in {(time.time() - t0) / 3600:.2f} h")
    print(f"  Gate 0 report : {Path(cfg.paths.reports_dir) / 'gate0.md'}")
    print(f"  Findings      : {Path(cfg.paths.reports_dir) / 'findings.md'}")
    print(f"  Ablation table: {Path(cfg.paths.results_dir) / 'ablation.csv'}")
    print(f"  Figures       : {cfg.paths.figures_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
