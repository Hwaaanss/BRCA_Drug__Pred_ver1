"""Typed, hierarchical configuration.

Every experimental choice lives here or in a YAML file under ``configs/``.
Nothing is hard-coded in model / training code.

Usage
-----
    cfg = load_config("configs/base.yaml", overrides=["train.epochs=5"])
    cfg.model.d_model            # -> 256
    cfg.to_dict()                # -> plain dict, JSON serialisable
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from functools import lru_cache
from typing import Any, Mapping, Sequence, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised on an unknown key, a bad type, or an invalid value."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    data_root: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    results_dir: str = "results"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"

    def ensure(self) -> None:
        for f in fields(self):
            Path(getattr(self, f.name)).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """GDSC / TCGA ingestion and feature construction."""

    # --- GDSC raw viability ---
    gdsc_versions: list[int] = field(default_factory=lambda: [2])
    neg_control_tags: list[str] = field(default_factory=lambda: ["NC-1", "NC-0"])
    pos_control_tags: list[str] = field(default_factory=lambda: ["B"])
    trim_viability: bool = True          # clip normalised viability to [0, 1]
    min_points_per_pair: int = 4
    max_points_per_pair: int = 16
    min_conc_um: float = 1e-6            # concentrations <= 0 are dropped (log)
    aggregate_replicates: bool = True    # average technical replicates per (pair, conc)

    # --- cell-line omics ---
    omics_modalities: list[str] = field(
        default_factory=lambda: ["expression", "mutation", "cnv"]
    )
    # Explicit file overrides; empty means "look under paths.raw_dir/omics".
    omics_files: dict[str, str] = field(default_factory=dict)
    expression_log1p: bool = True
    min_gene_variance: float = 0.0
    max_genes: int = 12000               # variance-filtered gene universe

    # --- pathway grouping / tokenisation ---
    pathway_gmt: str = "data/raw/pathways/ReactomePathways.gmt"
    extra_gmt: list[str] = field(default_factory=list)
    min_genes_per_group: int = 10
    max_genes_per_group: int = 128
    target_n_tokens: int = 400
    n_latent_tokens: int = 32            # for features outside any pathway

    # --- drugs ---
    drug_features: str = "fingerprint"   # fingerprint | onehot
    fingerprint_bits: int = 1024
    fingerprint_radius: int = 2
    drug_descriptor_file: str = "data/processed/drug_features.parquet"

    # --- histology ---
    histology_dir: str = "data/processed/uni_features"
    max_patches: int = 2048              # subsampled per slide (memory guard)
    histo_feature_dim: int = 1024        # UNI ViT-L/16

    # --- TCGA ---
    tcga_dir: str = "data/processed/tcga"
    clinical_cmax_csv: str = "data/clinical_cmax.csv"
    require_verified_cmax: bool = False

    def __post_init__(self) -> None:
        if self.drug_features not in {"fingerprint", "onehot"}:
            raise ConfigError(f"data.drug_features must be fingerprint|onehot, got {self.drug_features!r}")
        bad = set(self.omics_modalities) - {"expression", "mutation", "cnv", "proteomic"}
        if bad:
            raise ConfigError(f"unknown omics modalities: {sorted(bad)}")
        if self.min_points_per_pair < 3:
            raise ConfigError("data.min_points_per_pair must be >= 3 to identify a 3-parameter curve")


@dataclass
class SplitConfig:
    n_folds: int = 5
    seed: int = 42
    val_fraction: float = 0.15           # carved out of the training part of a fold
    schemes: list[str] = field(default_factory=lambda: ["LCO", "LPO", "LDO"])
    primary: str = "LCO"

    def __post_init__(self) -> None:
        from hill.constants import SPLIT_NAMES

        bad = set(self.schemes) - set(SPLIT_NAMES)
        if bad:
            raise ConfigError(f"unknown split schemes: {sorted(bad)}")
        if self.primary not in self.schemes:
            raise ConfigError(f"splits.primary={self.primary!r} not in splits.schemes")
        if self.n_folds < 2:
            raise ConfigError("splits.n_folds must be >= 2")


@dataclass
class HistologyConfig:
    enabled: bool = False
    feature_dim: int = 1024
    n_prototypes: int = 16               # K in ABMIL
    gate_hidden: int = 128
    gate_efficacy: bool = True           # gamma_E path exists
    gate_potency: bool = True            # gamma_m path exists
    train_gamma: bool = False            # Stage 0 keeps gamma frozen at 0


@dataclass
class ModelConfig:
    head: str = "curve"                  # curve (HILL) | scalar (ScalarHILL)
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    head_dropout: float = 0.1
    drug_hidden: int = 256
    token_dropout: float = 0.0
    init_e_logit: float = -2.0           # sigmoid(-2) ~= 0.12 initial E_inf
    init_slope_raw: float = 0.5
    histology: HistologyConfig = field(default_factory=HistologyConfig)

    def __post_init__(self) -> None:
        if self.head not in {"curve", "scalar"}:
            raise ConfigError(f"model.head must be curve|scalar, got {self.head!r}")
        if self.d_model % self.n_heads != 0:
            raise ConfigError(
                f"model.d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


@dataclass
class LossConfig:
    kind: str = "gaussian"               # gaussian | beta
    heteroscedastic: bool = True         # per-drug log sigma_j
    sigma_from_drug_features: bool = True  # required for LDO validity
    init_log_sigma: float = -2.0
    min_log_sigma: float = -6.0
    max_log_sigma: float = 2.0
    beta_init_log_phi: float = 2.0
    clinical_weight: float = 1.0
    source_weight_stage1: float = 0.5    # keep L_source during TCGA fine-tuning
    scalar_loss: str = "mse"             # ScalarHILL objective on ln IC50

    def __post_init__(self) -> None:
        if self.kind not in {"gaussian", "beta"}:
            raise ConfigError(f"loss.kind must be gaussian|beta, got {self.kind!r}")
        if self.scalar_loss not in {"mse", "huber"}:
            raise ConfigError(f"loss.scalar_loss must be mse|huber, got {self.scalar_loss!r}")


@dataclass
class TrainConfig:
    batch_size: int = 128                # pairs per batch (each carries K points)
    eval_batch_size: int = 256
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    patience: int = 10                   # early stopping on the validation primary metric
    min_epochs: int = 5
    amp_dtype: str = "bf16"              # bf16 | fp16 | none
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    log_every: int = 50
    stage1_epochs: int = 30
    stage1_lr: float = 1e-4
    deterministic: bool = False          # cudnn deterministic (slower)

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ConfigError(f"train.amp_dtype must be bf16|fp16|none, got {self.amp_dtype!r}")
        if self.patience < 1:
            raise ConfigError("train.patience must be >= 1")


@dataclass
class EvalConfig:
    min_pairs_per_drug: int = 5
    reference: str = "published"         # published | refit_m3
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0
    auc_n_grid: int = 256                # numeric cross-check grid for the closed form


@dataclass
class HPOConfig:
    n_trials: int = 10
    epochs: int = 25
    patience: int = 6
    fold: int = 0                        # single fold during search (cost control)
    timeout_hours: float | None = None
    storage: str = "results/hpo/optuna.db"
    study_name: str = "hill_lco"
    direction: str = "maximize"
    metric: str = "delta_pcc_vs_naive"
    seed: int = 42
    pruner_warmup_epochs: int = 8


@dataclass
class AblationConfig:
    seeds: int = 10
    fold_cycling: bool = True            # seed s evaluates fold (s % n_folds)
    full_cv: bool = False                # if True, every seed runs every fold
    steps: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    enforce_stop_rule: bool = True       # halt after step 0 if it loses to MOLI
    baselines: list[str] = field(
        default_factory=lambda: ["naive", "elasticnet", "randomforest", "moli", "superfelt"]
    )


@dataclass
class ResourceConfig:
    """Hard limits for the target machine (1x A100-80GB, 100 GB RAM, 10 cores, 30 GB disk)."""

    gpu_index: int = 0
    max_gpu_memory_fraction: float = 0.92
    torch_threads: int = 10
    max_ram_gb: float = 100.0
    max_disk_gb: float = 30.0
    ram_warn_fraction: float = 0.85


@dataclass
class Config:
    name: str = "hill_base"
    seed: int = 42
    device: str = "cuda"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # ---- serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in {".yaml", ".yml"}:
            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True), encoding="utf-8")
        else:
            p.write_text(self.to_json(), encoding="utf-8")

    def copy_with(self, overrides: Mapping[str, Any] | Sequence[str]) -> "Config":
        d = self.to_dict()
        _apply_overrides(d, overrides)
        return from_dict(Config, d)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _coerce(value: Any, target: Any, path: str) -> Any:
    """Coerce ``value`` to the annotated ``target`` type (best effort, strict)."""
    if is_dataclass(target):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected mapping for {target.__name__}, got {type(value).__name__}")
        return from_dict(target, value, path=path)

    origin = get_origin(target)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
        (inner,) = get_args(target) or (Any,)
        return [_coerce(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected mapping, got {type(value).__name__}")
        return dict(value)
    if origin is not None:  # Optional[...] / Union[...]
        args = [a for a in get_args(target) if a is not type(None)]  # noqa: E721
        if value is None:
            return None
        for a in args:
            try:
                return _coerce(value, a, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: cannot coerce {value!r} to {target}")

    if target is Any:
        return value
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in {"true", "1", "yes"}:
                return True
            if value.lower() in {"false", "0", "no"}:
                return False
        raise ConfigError(f"{path}: expected bool, got {value!r}")
    if target is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected float, got bool")
        return float(value)
    if target is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected int, got bool")
        if isinstance(value, float) and not float(value).is_integer():
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return int(value)
    if target is str:
        return str(value)
    return value


@lru_cache(maxsize=None)
def _resolved_hints(cls: type) -> dict[str, Any]:
    """``dataclasses.fields`` gives string annotations under PEP 563; resolve them."""
    return get_type_hints(cls)


def from_dict(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Build a dataclass from a mapping, rejecting unknown keys."""
    if not is_dataclass(cls):
        raise ConfigError(f"{cls} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ConfigError(f"{where}: unknown config key(s): {sorted(unknown)}")
    hints = _resolved_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        kwargs[name] = _coerce(data[name], hints[name], f"{path}.{name}" if path else name)
    return cls(**kwargs)  # type: ignore[return-value]


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_overrides(d: dict[str, Any], overrides: Mapping[str, Any] | Sequence[str]) -> None:
    """Apply ``a.b.c=value`` strings or a flat mapping of dotted keys, in place."""
    items: list[tuple[str, Any]]
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = []
        for raw in overrides:
            if "=" not in raw:
                raise ConfigError(f"override {raw!r} must look like key.subkey=value")
            k, v = raw.split("=", 1)
            items.append((k.strip(), _parse_scalar(v.strip())))

    for key, value in items:
        parts = key.split(".")
        node = d
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise ConfigError(f"override {key!r}: no config section {p!r}")
            node = node[p]
        if parts[-1] not in node:
            raise ConfigError(f"override {key!r}: unknown config key {parts[-1]!r}")
        node[parts[-1]] = value


def _deep_update(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Sequence[str] | Mapping[str, Any] | None = None,
    extra_files: Sequence[str | os.PathLike[str]] = (),
) -> Config:
    """Load ``configs/*.yaml``, layer ``extra_files``, then dotted overrides.

    Overrides are applied to the *complete* config (defaults included), so a key
    may be overridden even when the YAML file does not mention its section.
    """
    merged: dict[str, Any] = {}
    for p in ([path] if path else []) + list(extra_files):
        if p is None:
            continue
        raw = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{p}: top level of a config file must be a mapping")
        _deep_update(merged, raw)
    cfg = from_dict(Config, merged)
    if overrides:
        cfg = cfg.copy_with(overrides)
    return cfg
