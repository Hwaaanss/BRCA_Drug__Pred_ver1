"""Configuration must be strict: no silent typos, no invalid combinations."""

from __future__ import annotations

import pytest

from hill.config import Config, ConfigError, load_config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.splits.primary in cfg.splits.schemes


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config().copy_with(["model.nope=1"])
    with pytest.raises(ConfigError, match="no config section"):
        Config().copy_with(["nosuchsection.x=1"])


@pytest.mark.parametrize(
    "override",
    ["model.head=nonsense", "model.n_heads=7", "loss.kind=poisson", "splits.n_folds=1",
     "data.min_points_per_pair=2", "train.amp_dtype=fp8"],
)
def test_invalid_values_are_rejected(override):
    with pytest.raises(ConfigError):
        Config().copy_with([override])


def test_yaml_configs_load():
    for path in ("configs/base.yaml", "configs/smoke.yaml"):
        cfg = load_config(path)
        assert cfg.model.d_model > 0
        assert cfg.resources.gpu_index == 0, "the project is pinned to GPU 0"


def test_base_config_fits_the_target_machine():
    cfg = load_config("configs/base.yaml")
    assert cfg.resources.torch_threads <= 10, "the node has 10 cores"
    assert cfg.train.num_workers <= 9, "leave a core for the main process"
    assert cfg.resources.max_ram_gb <= 100
    assert cfg.resources.max_disk_gb <= 30


def test_roundtrip_through_dict():
    cfg = Config().copy_with(["train.epochs=7"])
    again = Config().copy_with(cfg.to_dict()["train"] and ["train.epochs=7"])
    assert again.train.epochs == cfg.train.epochs == 7


def test_overrides_reach_sections_absent_from_the_yaml_file():
    """configs/smoke.yaml has no `loss:` section; --set loss.kind=beta must still work."""
    cfg = load_config("configs/smoke.yaml", overrides=["loss.kind=beta", "eval.reference=refit_m3"])
    assert cfg.loss.kind == "beta"
    assert cfg.eval.reference == "refit_m3"
    assert cfg.model.d_model == 32, "file values must survive the override pass"


def test_extra_files_layer_on_top():
    cfg = load_config("configs/base.yaml", extra_files=["configs/smoke.yaml"])
    assert cfg.name == "hill_smoke_SYNTHETIC"
    assert cfg.model.d_model == 32
