"""Raw-plate normalisation must match gdscIC50's definition exactly (guide §2 G-1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hill.config import Config
from hill.data.gdsc import IngestQC, build_points_table, normalize_wells, points_to_padded


def _wells() -> pd.DataFrame:
    rows = []
    # one plate: negative control at 1000, blank at 100
    for _ in range(6):
        rows.append(dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=np.nan, DRUG_ID=np.nan,
                         CONC=np.nan, TAG="NC-1", INTENSITY=1000.0))
    for _ in range(3):
        rows.append(dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=np.nan, DRUG_ID=np.nan,
                         CONC=np.nan, TAG="B", INTENSITY=100.0))
    # treated wells at known viabilities
    for i, v in enumerate([1.0, 0.75, 0.5, 0.25, 0.0]):
        rows.append(dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=905947.0, DRUG_ID=1.0,
                         CONC=10.0 / 2**i, TAG=f"L1-D{i + 1}-S", INTENSITY=100.0 + 900.0 * v))
    # a combination well that must be dropped
    rows.append(dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=905947.0, DRUG_ID=1.0, CONC=10.0,
                     TAG="L1-D1-S+L2-D1-S", INTENSITY=500.0))
    return pd.DataFrame(rows)


def test_normalisation_matches_the_control_definition():
    qc = IngestQC()
    out = normalize_wells(_wells(), qc=qc)
    assert len(out) == 5, "combination wells must be dropped"
    assert qc.n_combination_wells_dropped == 1
    expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    assert np.allclose(np.sort(out["viability"].to_numpy())[::-1], expected, atol=1e-9)


def test_viability_is_trimmed_to_unit_interval():
    wells = _wells()
    wells.loc[len(wells)] = dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=905947.0, DRUG_ID=2.0,
                                 CONC=1.0, TAG="L2-D1-S", INTENSITY=5000.0)
    out = normalize_wells(wells, trim=True)
    assert out["viability"].max() <= 1.0 and out["viability"].min() >= 0.0


def test_pipeline_produces_pair_and_point_tables(tmp_path):
    path = tmp_path / "GDSC2_public_raw_data_TEST.csv"
    _wells().to_csv(path, index=False)
    cfg = Config().copy_with(["data.min_points_per_pair=4"])
    points, pairs, qc = build_points_table({"GDSC2": path}, cfg)
    assert len(pairs) == 1 and len(points) == 5
    assert pairs["n_points"].iloc[0] == 5
    assert np.isclose(pairs["max_conc"].iloc[0], 10.0)
    assert np.allclose(points["log_conc"], np.log(points["conc"]))

    padded = points_to_padded(points, pairs, max_points=8)
    assert padded["log_conc"].shape == (1, 8)
    assert padded["mask"].sum() == 5
    # padded entries must be masked out, never silently read as zeros
    assert not padded["mask"][0, 5:].any()


def test_pairs_below_the_point_threshold_are_dropped(tmp_path):
    path = tmp_path / "GDSC2_public_raw_data_TEST.csv"
    _wells().to_csv(path, index=False)
    cfg = Config().copy_with(["data.min_points_per_pair=6"])
    points, pairs, qc = build_points_table({"GDSC2": path}, cfg)
    assert len(pairs) == 0
    assert qc.n_pairs_dropped_too_few_points == 1


def test_gdsc2_supersedes_gdsc1_for_the_same_pair(tmp_path):
    """When both releases screened a pair, only the newer release's points survive."""
    import pandas as pd

    from hill.config import Config
    from hill.data.gdsc import build_points_table

    def wells(intensity_scale: float) -> pd.DataFrame:
        rows = [dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=float("nan"), DRUG_ID=float("nan"),
                     CONC=float("nan"), TAG="NC-1", INTENSITY=1000.0) for _ in range(4)]
        rows += [dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=float("nan"), DRUG_ID=float("nan"),
                      CONC=float("nan"), TAG="B", INTENSITY=100.0) for _ in range(2)]
        for i in range(5):
            rows.append(dict(BARCODE="P1", SCAN_ID="S1", COSMIC_ID=1.0, DRUG_ID=1.0,
                             CONC=10.0 / 2**i, TAG=f"L1-D{i + 1}-S",
                             INTENSITY=100.0 + 900.0 * intensity_scale))
        return pd.DataFrame(rows)

    p1 = tmp_path / "GDSC1_public_raw_data_TEST.csv"
    p2 = tmp_path / "GDSC2_public_raw_data_TEST.csv"
    wells(0.9).to_csv(p1, index=False)
    wells(0.2).to_csv(p2, index=False)

    cfg = Config().copy_with(["data.gdsc_versions=[1, 2]"])
    points, pairs, _ = build_points_table({"GDSC1": p1, "GDSC2": p2}, cfg)
    assert len(pairs) == 1
    assert set(points["source"]) == {"GDSC2"}, "GDSC2 must win for a pair screened in both"
    assert abs(float(points["viability"].iloc[0]) - 0.2) < 1e-6
