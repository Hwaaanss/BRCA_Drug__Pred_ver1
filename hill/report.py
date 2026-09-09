"""Write ``reports/findings.md`` from whatever results exist.

Deliverable §8.4.  Negative results are written down exactly as they came out:
if the ladder stopped, if a gate failed, if the model lost to a baseline, the
report says so (project rules R2 and R4).

    python -m hill.report --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hill.config import Config, load_config
from hill.data.synthetic import is_synthetic
from hill.utils.logging import get_logger
from hill.utils.provenance import git_hash
from hill.utils.stats import paired_test

log = get_logger("report")


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(x: Any, d: int = 4) -> str:
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.{d}f}"
    return str(x)


def build_findings(cfg: Config) -> str:
    res = Path(cfg.paths.results_dir)
    g1 = _load_json(res / "gate0" / "g1_raw_access.json")
    g2 = _load_json(res / "gate0" / "g2_censoring.json")
    g3 = _load_json(res / "gate0" / "g3_refit.json")
    g4 = _load_json(res / "gate0" / "g4_variance.json")
    g5 = _load_json(res / "gate0" / "g5_availability.json")
    hpo = _load_json(Path(cfg.hpo.storage).parent / "best_params.json")
    stop = _load_json(res / "ablation_stop_rule.json")
    ablation = pd.read_csv(res / "ablation.csv") if (res / "ablation.csv").exists() else None
    lrt_files = sorted((res / "stage1").glob("lrt_seed*.json"))
    lrt = _load_json(lrt_files[0]) if lrt_files else None

    synth = is_synthetic(cfg.paths.data_root)
    lines: list[str] = ["# Findings", ""]
    if synth:
        lines += ["> **SYNTHETIC DATA — NOT A SCIENTIFIC RESULT.** These numbers come from "
                  "simulated data produced by `hill.data.synthetic`.", ""]
    lines += [
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- git: `{git_hash()}`",
        f"- config: `{getattr(cfg, '_source_path', 'configs/base.yaml')}`",
        "",
        "## 1. What the data says before any model (Gate 0)",
        "",
    ]
    if g1:
        lines.append(
            f"- **G-1 {'PASSED' if g1.get('passed') else 'FAILED'}** — our own normalisation plus a "
            f"two-parameter refit reproduces the published LN_IC50 with r = "
            f"{_fmt(g1.get('ic50_pearson_vs_published'), 3)} over "
            f"{_fmt(g1.get('n_pairs_audited'))} pairs "
            f"(`python -m hill.audit.raw_access`)."
        )
    if g2:
        lines.append(
            f"- **G-2** — {_fmt(100 * g2.get('censored_fraction_overall', float('nan')), 1)}% of pairs "
            f"carry a censored IC50 (beyond the maximum tested dose) "
            f"(`python -m hill.audit.censoring`)."
        )
        if g2.get("censored_fraction_overall", 0) < 0.10:
            lines.append(
                "  - Below 10%: the censoring argument is weaker on this release than the design "
                "assumed, and the report says so rather than hiding it."
            )
    if g3:
        lines.append(
            f"- **G-3 {'SUPPORTED' if g3.get('premise_supported') else 'NOT SUPPORTED'}** — the "
            f"free-floor model is preferred by BIC on "
            f"{_fmt(100 * g3.get('fraction_M3_preferred_BIC', float('nan')), 1)}% of pairs "
            f"(Wilcoxon p = {_fmt(g3.get('wilcoxon_p_M3_better'), 3)}); fitted E_inf median "
            f"{_fmt(g3.get('e_inf_distribution', {}).get('median'), 3)}, "
            f"{_fmt(100 * g3.get('fraction_e_inf_ge_0.5', float('nan')), 1)}% of pairs have no IC50 at all "
            f"(`python -m hill.audit.refit`)."
        )
        if not g3.get("premise_supported"):
            lines.append(
                "  - **This is a negative result for the project's premise.** The three-parameter "
                "model does not clearly beat the official fit on this data; the design's central "
                "assumption is not supported here."
            )
    if g4:
        lines.append(
            f"- **G-4** — the drug main effect explains f_drug = {_fmt(g4.get('f_drug'), 3)} of label "
            "variance, which is why drug-pooled correlation is not reported anywhere in this project."
        )
    if g5:
        lines.append(
            f"- **G-5** — {g5.get('n_patients_omics_and_slide', 'n/a')} TCGA patients have both omics "
            f"and slide features; {g5.get('n_evaluable_outcomes', 'n/a')} treatment records have an "
            f"evaluable outcome; "
            f"{g5.get('cmax', {}).get('n_with_value', 'n/a')} of "
            f"{g5.get('cmax', {}).get('n_drugs_listed', 'n/a')} drugs have a clinical Cmax "
            f"({g5.get('cmax', {}).get('n_verified', 0)} human-verified)."
        )
        missing = g5.get("cmax", {}).get("drugs_without_cmax", [])
        if missing:
            lines.append(f"  - excluded from the clinical likelihood: {', '.join(map(str, missing))}")

    lines += ["", "## 2. Hyper-parameter search", ""]
    if hpo:
        lines += [
            f"- {hpo['n_trials_completed']} trials completed, {hpo['n_trials_pruned']} pruned, "
            f"optimising `{hpo['metric']}` on fold {hpo['search_fold']} with a "
            f"{hpo['search_epochs']}-epoch budget.",
            f"- best value: {_fmt(hpo['best_value'])}",
            "- search space and every sampled value: `results/hpo/trials.csv`",
            "",
            "| parameter | value |", "|---|---|",
        ]
        lines += [f"| `{k}` | {v} |" for k, v in hpo["best_params"].items()]
    else:
        lines.append("- not run.")

    lines += ["", "## 3. Ablation ladder", ""]
    if ablation is not None and not ablation.empty:
        metric = "delta_pcc_vs_naive"
        summary = (
            ablation[np.isfinite(ablation.get(metric, np.nan))]
            .groupby(["step", "method"], dropna=False)[metric]
            .agg(["mean", "std", "count"]).reset_index().sort_values(["step", "method"])
        )
        lines += ["| step | method | dPCC vs naive (mean ± sd) | runs |", "|---|---|---|---|"]
        for _, r in summary.iterrows():
            sd = "" if not np.isfinite(r["std"]) else f" ± {r['std']:.4f}"
            lines.append(f"| {int(r['step'])} | {r['method']} | {r['mean']:.4f}{sd} | {int(r['count'])} |")

        step0 = ablation[ablation["method"] == "step0"][metric].dropna()
        step2 = ablation[ablation["method"] == "step2"][metric].dropna()
        if len(step0) and len(step2):
            n = min(len(step0), len(step2))
            test = paired_test(step2.to_numpy()[:n], step0.to_numpy()[:n])
            verdict = "beats" if test["mean_diff"] > 0 else "does NOT beat"
            lines += [
                "",
                f"**HILL vs ScalarHILL (the project's central claim):** the curve head {verdict} the "
                f"scalar head by {test['mean_diff']:+.4f} dPCC over {n} paired runs "
                f"(Wilcoxon p = {_fmt(test['wilcoxon_p'], 4)}, Cohen's d = {_fmt(test['cohens_d'], 2)}).",
            ]
            if test["mean_diff"] <= 0:
                lines.append(
                    "  - **Negative result.** On this data the curve formulation did not improve the "
                    "derived-IC50 correlation. Reported as it came out."
                )
        if "clinical_roc_auc" in ablation.columns and np.isfinite(ablation["clinical_roc_auc"]).any():
            clin = ablation.groupby("step")["clinical_roc_auc"].agg(["mean", "std", "count"]).dropna()
            lines += ["", "| step | clinical ROC-AUC (mean ± sd) | runs |", "|---|---|---|"]
            for step, r in clin.iterrows():
                sd = "" if not np.isfinite(r["std"]) else f" ± {r['std']:.3f}"
                lines.append(f"| {int(step)} | {r['mean']:.3f}{sd} | {int(r['count'])} |")
    else:
        lines.append("- not run.")

    if stop:
        lines += ["", "### Stop rule (step 0 must beat MOLI)", ""]
        lines.append(
            f"- triggered: **{stop.get('triggered')}** — step 0 mean {_fmt(stop.get('step0_mean'))} vs "
            f"MOLI {_fmt(stop.get('moli_mean'))} over {stop.get('n_seeds', 'n/a')} seed(s)."
        )
        if stop.get("triggered"):
            lines.append(
                "  - The ladder was halted here on purpose: a curve head on top of an encoder that "
                "cannot beat MOLI would make the cause impossible to isolate."
            )

    lines += ["", "## 4. Histology gates", ""]
    if lrt:
        lines += [
            f"- gamma_E: lambda = {_fmt(lrt['lrt_gamma_E']['lambda'], 2)}, "
            f"p (boundary) = {_fmt(lrt['lrt_gamma_E']['p_boundary'])}, "
            f"p (chi2_1) = {_fmt(lrt['lrt_gamma_E']['p_chi2'])}",
            f"- gamma_m: lambda = {_fmt(lrt['lrt_gamma_m']['lambda'], 2)}, "
            f"p (boundary) = {_fmt(lrt['lrt_gamma_m']['p_boundary'])}, "
            f"p (chi2_1) = {_fmt(lrt['lrt_gamma_m']['p_chi2'])}",
            "",
            "The two models are exactly nested by construction (`tests/test_gamma.py` asserts "
            "bit-identical outputs at gamma = 0), which is what makes this test admissible. Note that "
            "the parameters are estimated by SGD on a deep model, so the chi-square reference "
            "distribution is an approximation and is reported as such.",
        ]
    else:
        lines.append("- not run (`python -m hill.stage1 --lrt`).")

    lines += [
        "", "## 5. What did not work", "",
        "Fill this section in by hand as the run progresses — it is meant to record the things that "
        "were tried and failed, not only the final numbers. The automated sections above already "
        "report negative outcomes where they occurred.",
        "", "## 6. Reproduction", "",
        "```bash",
        "python -m hill.data.prepare   --config configs/base.yaml",
        "python -m hill.audit.run_gate0 --config configs/base.yaml",
        "python -m hill.hpo            --config configs/base.yaml --n-trials 10",
        f"python -m hill.ablation       --config configs/base.yaml --extra {Path(cfg.hpo.storage).parent / 'best_params.yaml'} --seeds 10",
        "python -m hill.figures.run_all --config configs/base.yaml",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write reports/findings.md")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    setattr(cfg, "_source_path", args.config)
    cfg.paths.ensure()
    out = Path(cfg.paths.reports_dir) / "findings.md"
    out.write_text(build_findings(cfg), encoding="utf-8")
    log.info("wrote %s", out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
