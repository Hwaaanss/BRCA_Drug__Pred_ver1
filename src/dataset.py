"""
Multi-modal dataset for PathOmicDRP.

Handles three training scenarios:
1. GDSC cell-line mode: predict IC50 from omics (no histology)
2. TCGA patient mode: predict clinical response from omics + optional histology
3. Transfer mode: oncoPredict-imputed IC50 as target for TCGA patients
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class PathOmicDataset(Dataset):
    """Multi-modal dataset combining genomic, transcriptomic, proteomic features."""

    def __init__(
        self,
        patient_ids: list,
        genomic_df: pd.DataFrame,
        transcriptomic_df: pd.DataFrame,
        proteomic_df: pd.DataFrame,
        targets: pd.Series = None,
        histology_dir: str = None,
        scalers: dict = None,
        fit_scalers: bool = False,
    ):
        self.patient_ids = list(patient_ids)
        self.targets = targets
        self.histology_dir = histology_dir

        # Index dataframes by patient_id
        self.genomic = genomic_df.set_index('patient_id') if 'patient_id' in genomic_df.columns else genomic_df
        self.transcriptomic = transcriptomic_df.set_index('patient_id') if 'patient_id' in transcriptomic_df.columns else transcriptomic_df
        self.proteomic = proteomic_df.set_index('patient_id') if 'patient_id' in proteomic_df.columns else proteomic_df

        # Feature columns (exclude patient_id if still present)
        self.gen_cols = [c for c in self.genomic.columns if c != 'patient_id']
        self.trans_cols = [c for c in self.transcriptomic.columns if c != 'patient_id']
        self.prot_cols = [c for c in self.proteomic.columns if c != 'patient_id']

        # Scaling
        if scalers is not None:
            self.scalers = scalers
        elif fit_scalers:
            self.scalers = self._fit_scalers()
        else:
            self.scalers = None

    def _fit_scalers(self) -> dict:
        scalers = {}
        avail = self.genomic.index.intersection(self.patient_ids)

        # Genomic: leave TMB for scaling, mutations are binary
        gen_data = self.genomic.loc[avail, self.gen_cols].values.astype(np.float32)
        s = StandardScaler()
        s.fit(gen_data)
        scalers['genomic'] = s

        # Transcriptomic: log1p + standard scale
        avail_t = self.transcriptomic.index.intersection(self.patient_ids)
        trans_data = np.log1p(self.transcriptomic.loc[avail_t, self.trans_cols].values.astype(np.float32))
        s = StandardScaler()
        s.fit(trans_data)
        scalers['transcriptomic'] = s

        # Proteomic: standard scale
        avail_p = self.proteomic.index.intersection(self.patient_ids)
        prot_data = self.proteomic.loc[avail_p, self.prot_cols].values.astype(np.float32)
        s = StandardScaler()
        s.fit(prot_data)
        scalers['proteomic'] = s

        return scalers

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # Genomic
        if pid in self.genomic.index:
            gen = self.genomic.loc[pid, self.gen_cols].values.astype(np.float32)
            gen_mask = True
        else:
            gen = np.zeros(len(self.gen_cols), dtype=np.float32)
            gen_mask = False

        # Transcriptomic
        if pid in self.transcriptomic.index:
            trans = self.transcriptomic.loc[pid, self.trans_cols].values.astype(np.float32)
            trans = np.log1p(np.maximum(trans, 0))  # log1p transform TPM
            trans_mask = True
        else:
            trans = np.zeros(len(self.trans_cols), dtype=np.float32)
            trans_mask = False

        # Proteomic
        if pid in self.proteomic.index:
            prot = self.proteomic.loc[pid, self.prot_cols].values.astype(np.float32)
            prot_mask = True
        else:
            prot = np.zeros(len(self.prot_cols), dtype=np.float32)
            prot_mask = False

        # Apply scaling
        if self.scalers:
            if gen_mask and 'genomic' in self.scalers:
                gen = self.scalers['genomic'].transform(gen.reshape(1, -1)).flatten()
            if trans_mask and 'transcriptomic' in self.scalers:
                trans = self.scalers['transcriptomic'].transform(trans.reshape(1, -1)).flatten()
            if prot_mask and 'proteomic' in self.scalers:
                prot = self.scalers['proteomic'].transform(prot.reshape(1, -1)).flatten()

        # Histology features (pre-extracted .pt files)
        histo = None
        if self.histology_dir and os.path.exists(os.path.join(self.histology_dir, f"{pid}.pt")):
            histo = torch.load(os.path.join(self.histology_dir, f"{pid}.pt"), weights_only=True)

        sample = {
            'patient_id': pid,
            'genomic': torch.tensor(gen, dtype=torch.float32),
            'transcriptomic': torch.tensor(trans, dtype=torch.float32),
            'proteomic': torch.tensor(prot, dtype=torch.float32),
            'modality_mask': torch.tensor([gen_mask, trans_mask, prot_mask], dtype=torch.bool),
        }

        if histo is not None:
            sample['histology'] = histo

        if self.targets is not None and pid in self.targets.index:
            sample['target'] = torch.tensor(self.targets.loc[pid], dtype=torch.float32)

        return sample


def collate_fn(batch):
    """Custom collate function handling variable-size histology features."""
    result = {
        'patient_id': [s['patient_id'] for s in batch],
        'genomic': torch.stack([s['genomic'] for s in batch]),
        'transcriptomic': torch.stack([s['transcriptomic'] for s in batch]),
        'proteomic': torch.stack([s['proteomic'] for s in batch]),
        'modality_mask': torch.stack([s['modality_mask'] for s in batch]),
    }

    if 'target' in batch[0]:
        result['target'] = torch.stack([s['target'] for s in batch])

    # Handle histology (variable number of patches)
    has_histo = [s for s in batch if 'histology' in s]
    if has_histo:
        max_patches = max(s['histology'].shape[0] for s in has_histo)
        feat_dim = has_histo[0]['histology'].shape[1]
        histo_tensor = torch.zeros(len(batch), max_patches, feat_dim)
        histo_mask = torch.zeros(len(batch), max_patches, dtype=torch.bool)
        for i, s in enumerate(batch):
            if 'histology' in s:
                n = s['histology'].shape[0]
                histo_tensor[i, :n] = s['histology']
                histo_mask[i, :n] = True
        result['histology'] = histo_tensor
        result['histo_mask'] = histo_mask

    return result


def load_data(base_dir: str | os.PathLike | None = None):
    """Load all feature matrices and return DataFrames."""
    if base_dir is None:
        project_root = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
        data_root = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", project_root / "data")).resolve()
        base_dir = data_root / "07_integrated"
    genomic = pd.read_csv(os.path.join(base_dir, "X_genomic.csv"))
    transcriptomic = pd.read_csv(os.path.join(base_dir, "X_transcriptomic.csv"))
    proteomic = pd.read_csv(os.path.join(base_dir, "X_proteomic.csv"))
    master = pd.read_csv(os.path.join(base_dir, "sample_master_table.csv"))

    return {
        'genomic': genomic,
        'transcriptomic': transcriptomic,
        'proteomic': proteomic,
        'master': master,
    }
