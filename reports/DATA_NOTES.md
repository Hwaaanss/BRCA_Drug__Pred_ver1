# Data notes

What each external file is, where it comes from, and what is *not* guaranteed.

## GDSC

| file | used for | notes |
|---|---|---|
| `GDSC{1,2}_public_raw_data*.csv` | **the training signal**: one row = one plate well | ~1.2 GB per release uncompressed. Columns used: `BARCODE`, `SCAN_ID`, `COSMIC_ID`, `DRUG_ID`, `CONC`, `TAG`, `INTENSITY`/`FLUORESCENCE`. |
| `GDSC{1,2}_fitted_dose_response*.xlsx` | **evaluation only** — never a training target | supplies `LN_IC50`, `AUC`, `MAX_CONC` and therefore the censoring flag. |
| `Cell_Lines_Details.xlsx` | identifier harmonisation | maps cell-line names / model ids to COSMIC ids. |
| `screened_compounds*.csv` | drug names, targets, target pathway | drug ids in `data/clinical_cmax.csv` refer to this file. |
| `Cell_line_RMA_proc_basalExp.txt` | expression | genes × `DATA.<COSMIC_ID>` layout. |
| `mutations_all*.csv` | mutation | long format, reduced to a binary matrix. |
| `cnv_gistic*.csv` | copy number | orientation auto-detected. |

Normalisation follows `gdscIC50::normalizeData`:
`viability = (intensity − mean(blank)) / (mean(negative control) − mean(blank))`,
computed **per plate** (`BARCODE`, `SCAN_ID`) and trimmed to [0, 1].
Control tags are configurable (`data.neg_control_tags`, `data.pos_control_tags`);
the defaults are `NC-1`/`NC-0` for the negative control and `B` (blank) for the
positive control. Combination wells (a TAG naming two library positions) are
dropped and counted in the ingest QC.

**Filenames drift between GDSC releases.** All URLs live in
`configs/data_sources.yaml`; a failed download names the file and the URL and the
pipeline refuses to continue rather than using something else.

## TCGA

Downloaded from the GDC API by `hill/data/download_tcga.py`: clinical cases with
their treatment records, STAR-Counts expression (primary tumour only), and
open-access masked somatic mutations.

Response labels come from `treatment_outcome`: Complete/Partial Response → 1,
Stable/Progressive Disease → 0; everything else is dropped and counted.

**Whole-slide images are not downloaded.** TCGA-BRCA slides are ~600 GB and the
target node has 30 GB. The pipeline expects *pre-extracted* UNI patch features at
`data/processed/uni_features/<patient_id>.npy`, float32 `(n_patches, 1024)`,
subsampled to `data.max_patches` at load time. Extract them on a machine with the
storage, then copy only the feature files across.

## Clinical Cmax — read this before using Stage 1

`data/clinical_cmax.csv` ships with **`verified=false` on every row**. The values
are order-of-magnitude literature look-ups keyed to
Liston & Davis, *Clin Cancer Res* 2017;23(14):3489–3498. Nobody has checked them
against that source in this repository.

* Check each value, then set `verified=true`.
* Run with `data.require_verified_cmax=true` to use only checked values.
* Drugs with an empty `cmax_um` (OSI-027, Daporinad, ABT-737, AZD5991) are
  excluded from the clinical likelihood and listed in the G-5 audit. That is the
  correct state for a compound with no human pharmacokinetics.
* `drug_id` refers to the GDSC `screened_compounds.csv` of the release you
  downloaded — confirm the ids before a production run.

## Synthetic data

`hill/data/synthetic.py` writes a miniature but structurally faithful copy of
every input file above, for the smoke test only. It drops a `SYNTHETIC` marker in
the data root; reports get a banner, figures get a watermark and an
`_SYNTHETIC` filename suffix. No number produced from it is a scientific result.
