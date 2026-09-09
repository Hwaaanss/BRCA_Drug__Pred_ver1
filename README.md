# HILL — predicting dose–response *functions*, not IC50 numbers

HILL predicts a **dose–response curve** for every (sample, drug) pair and trains
on **raw, control-normalised viability measurements**. IC50, AUC and Emax are
*derived* from the predicted curve rather than regressed onto.

```
r_ij(c) = E_inf + (1 − E_inf) · sigmoid( −s · (log c − m) )
              ▲                  ▲              ▲
        efficacy ceiling     Hill slope     potency (log midpoint)
```

## Why

GDSC's official curve-fitting pipeline (`gdscIC50::logist3`) fits a **two**-parameter
logistic with the floor **pinned at 0** — it assumes every drug kills 100 % of cells
at a high enough dose. Three consequences follow, and they are all the same problem:

1. **Efficacy variation cannot be expressed.** The published fit has no Emax
   parameter, so it cannot be read out of the released files at all.
2. **That is where extrapolated IC50s come from.** When the true floor is high, a
   floor-at-zero fit pushes the midpoint past the highest tested dose. The
   censoring problem and the misspecification problem are one problem.
3. **Potency and efficacy are conflated.** A drug that kills 95 % at 1 µM and one
   that kills 55 % at 1 µM get the same label.

Every deep DRP model — DeepCDR, MOLI, SuperFELT, DRPreter, TransCDR, GBD-DRP —
regresses onto labels produced by that fit, so the field inherits the
misspecification. Training on the raw measurements instead:

* raises the supervision from 1 number to K ≈ 5–9 measurements per pair;
* **dissolves the censoring problem** — nothing is regressed onto a censored
  quantity, so no Tobit likelihood is needed, and a curve that never crosses 50 %
  is still fully identified by its observed points;
* predicts potency (`m`) and efficacy (`E_inf`) as separate quantities;
* makes monotonicity in dose a **structural** property of the parameterisation,
  not something the optimiser has to discover.

**The encoder is not the contribution.** Pathway tokenisation (SurvPath,
Pathformer, DRPreter) and drug-conditioned pathway attention (DRPreter) are reused
and cited. The contribution is the output head and the likelihood.

## Repository layout

```
hill/
  config.py           typed, strict configuration (no hyper-parameter is hard-coded)
  derive.py         ★ curve -> IC50 / AUC / Emax  (IC50 returns None when it does not exist)
  losses.py         ★ viability likelihood (Gaussian / Beta) + clinical likelihood at Cmax
  train.py            Stage 0 training loop, early stopping on the validation primary metric
  stage1.py           TCGA fine-tuning, histology gates, likelihood-ratio tests
  evaluate.py         per-drug metrics; drug-pooled correlation is refused by design
  hpo.py              Optuna search (10 trials), search space declared in one place
  ablation.py         the §7.4 ladder + baselines over 10 seeds
  report.py           writes reports/findings.md, negative results included
  run_all.py          the whole study in one command
  smoke.py            end-to-end run on SYNTHETIC data, then deletes it
  data/               GDSC raw wells, omics, drug features, tokenisation, splits, TCGA
  models/             curve_head ★, encoder, drug, histology gates, baselines
  audit/              Gate 0: G-1 … G-5 and the batched curve fitter
  figures/            every manuscript figure (PDF + PNG, 400 dpi, colourblind-safe)
configs/              base.yaml (production) · smoke.yaml (tiny) · data_sources.yaml (URLs)
tests/                70 tests, including test_gamma_zero_equivalence
```

## Gate 0 — the audit that runs before any model

`reports/gate0.md` is produced by `python -m hill.audit.run_gate0` and answers:

| gate | question | halts the project? |
|---|---|---|
| G-1 | Can we read raw viability and does our normalisation reproduce the published fits? | **yes** if it fails |
| G-2 | What fraction of published IC50s is censored? | no, reported either way |
| G-3 | Does a free-floor model beat the official two-parameter fit? | **yes** if not supported |
| G-4 | How much label variance is the drug main effect? | no, it justifies the metric choice |
| G-5 | What is actually available on the TCGA side? | no |

G-3 is the decisive experiment and uses no deep learning at all: both models are
fitted to the same points, then compared on residuals, AIC/BIC, the distribution
of the fitted `E_inf`, and the overlap between high-`E_inf` and censored pairs.

## Metrics

| metric | where | note |
|---|---|---|
| viability RMSE | per measurement | curve quality |
| per-drug PCC of derived IC50 | per drug, then averaged | comparison with prior work |
| **ΔPCC vs `NaiveMeanEffects`** | per drug | **primary metric** (DrEval-style normalisation) |
| Emax correlation | per drug | only this formulation can produce it |
| clinical ROC-AUC | TCGA, evaluated at Cmax | target domain |
| γ_E, γ_m + LRT | — | does morphology act on efficacy but not potency? |

**Drug-pooled ("global") PCC is not computed.** With a large drug main effect it
mostly measures which drug a row belongs to; `hill/evaluate.py` raises
`GlobalPCCForbiddenError` if it is requested.

Pairs whose predicted curve never reaches 50 % have **no IC50**. That is a result,
not a missing value: `derive.ln_ic50` returns NaN/None, and every metric is
reported both excluding and including those pairs, with the excluded fraction.

## Ablation ladder (one change at a time)

| step | configuration | question | stop rule |
|---|---|---|---|
| 0 | `ScalarHILL` (same encoder, MSE on ln IC50) | is the encoder sound? | **must beat MOLI or the ladder halts** |
| 1 | HILL curve head, single σ | what does the curve head add? | |
| 2 | + heteroscedastic σ_j | what does the likelihood's shape add? | |
| 3 | + TCGA transfer, γ = 0 | domain transfer | |
| 4 | + γ_E released | LRT on efficacy | |
| 5 | + γ_m released | does morphology touch potency too? | |

`ScalarHILL` shares the encoder, the data and the splits with HILL exactly, so the
HILL − ScalarHILL gap isolates the head and the loss. That comparison *is* the paper.

---

# Running it

## 1. Get the code

```bash
git clone https://github.com/Hwaaanss/PathOmicDPR_ver1.git
cd PathOmicDPR_ver1
git checkout claude/research-code-restructure-x9tkfp
# already cloned?  ->  git pull origin claude/research-code-restructure-x9tkfp
```

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate hill
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Install libraries and download the data

```bash
pip install -r requirements.txt          # no-op if environment.yml already ran
python -m hill.data.download --what all  # GDSC raw + fitted + omics + Reactome + annotation
python -m hill.data.download_tcga --project TCGA-BRCA --what clinical expression mutation
```

`configs/data_sources.yaml` holds every URL. GDSC filenames change between
releases: if a download fails, the error names the file and the URL to fix — the
pipeline never substitutes a different file silently.

Whole-slide images are **not** downloaded (≈ 600 GB, far beyond the 30 GB budget).
The pipeline consumes pre-extracted UNI patch features from
`data/processed/uni_features/<patient_id>.npy` (~4 MB per slide). Patients without
one are handled by the `no_histology` embedding and contribute through γ = 0.

## 4. Smoke test (a few minutes, deletes its own output)

```bash
python -m hill.smoke --with-tests
```

Runs the unit tests, then the whole pipeline — ingest → Gate 0 → HPO → ablation →
Stage 1 → figures → report — on tiny **SYNTHETIC** data, then removes everything
it created. Add `--keep` to inspect the artefacts.

## 5. The full study, one command

```bash
CUDA_VISIBLE_DEVICES=0 python -m hill.run_all --config configs/base.yaml --skip-download
```

Stages: prepare → Gate 0 → tests → HPO (10 trials) → ablation ladder (10 seeds,
best hyper-parameters) → Stage 1 + LRT → figures → `reports/findings.md`.
Gate 0 halts the run if G-1 fails or G-3 does not support the premise — that is
deliberate.

Outputs:

```
reports/gate0.md          the data audit, with a reproduction command per number
reports/findings.md       what worked and what did not, negative results included
reports/figures/*.pdf     Fig1 … Fig10 + supplementary, 400 dpi, PDF and PNG
results/ablation.csv      every run; ablation_summary.csv has mean ± sd over seeds
results/hpo/trials.csv    every trial, its sampled values and its score
results/checkpoints/      model weights + the config and scaler that produced them
results/logs/*.jsonl      structured per-epoch logs
```

## Resource budget (1× A100-80GB, 100 GB RAM, 10 cores, 30 GB disk)

`configs/base.yaml` is sized for exactly that node and `hill/utils/resources.py`
enforces it: the process is pinned to GPU 0, thread counts are capped at 10, the
CUDA allocator is capped at 92 % of the card, and a run refuses to start with less
than 2 GB of free disk.

| knob | value | why |
|---|---|---|
| `train.batch_size` | 256 pairs | 256 × ~7 points × ~400 tokens fits in 80 GB with bf16 |
| `train.amp_dtype` | `bf16` | A100 native; no loss scaling needed |
| `train.num_workers` | 8 | 8 loaders + main + CUDA feeder ≈ 10 cores |
| `data.max_patches` | 2048 | caps slide memory at ~4 MB per patient |
| `ablation.fold_cycling` | `true` | seed *s* runs fold *s* mod 5: 10 seeds cover all 5 folds twice at 1/5 the cost |

Rough wall-clock for the full run: **~20 h** (HPO ≈ 2 h, ladder ≈ 15–18 h).
Set `ablation.full_cv=true` for the complete seed × fold grid (5× longer), or lower
`--seeds` / `--n-trials` to trade precision for time.

## Honest-reporting rules baked into the code

* **Synthetic data is unmistakable.** `hill.data.synthetic` drops a `SYNTHETIC`
  marker; reports get a banner, figures get a watermark, filenames get a suffix.
* **A failure is reported as a failure.** Downloads, gates and the stop rule return
  non-zero and say what went wrong; nothing is silently substituted.
* **Every number has a reproduction command** printed next to it in the reports.
* **Cmax values ship unverified.** `data/clinical_cmax.csv` carries a
  `verified` flag that is `false` for every literature look-up. Check each against
  its cited source and flip the flag; run with `data.require_verified_cmax=true`
  to use only checked values. Drugs with no Cmax are excluded and listed.

## Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is False | the pip wheel's CUDA build does not match the driver — reinstall torch from the matching `--index-url` (cu121 for driver < 550) |
| `no raw viability file for GDSC2` | the release filename changed; update `configs/data_sources.yaml` or drop the CSV into `data/raw/gdsc_raw/` |
| `no cell line is present in every modality` | identifier harmonisation failed; check that `Cell_Lines_Details.xlsx` is present so names map to COSMIC ids |
| `no gene set matched the feature universe` | the GMT uses a different symbol namespace than the omics matrix |
| RDKit missing → hashed n-gram fingerprints | install `rdkit` from conda-forge; LDO results are weaker without it |
| out of GPU memory | halve `train.batch_size`, then `train.eval_batch_size`; lower `data.max_patches` |
| out of disk | delete `results/checkpoints/`; drop `data.gdsc_versions` to `[2]` |

## Citation

The encoder follows SurvPath (CVPR 2024), Pathformer (Bioinformatics 2024) and
DRPreter (IJMS 2022); the naive-baseline normalisation follows DrEval (Nature
Communications 2026); using Cmax as the clinical-relevance anchor is standard
pharmacology practice, not a contribution of this work.

## License

MIT — see `LICENSE`.
