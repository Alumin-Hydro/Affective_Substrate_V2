"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.schema import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    assert isinstance(config, AppConfig)
    assert config.substrate.k == 4
    assert config.gates.lyapunov.lambda_min == 0.01


def test_load_phase0_config() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    assert config.substrate.noise_sigma == 0.0
    assert config.substrate.ic_mode == "sane"


def test_invalid_gate_threshold_rejected() -> None:
    data = load_config(ROOT / "config" / "phase0.yaml").model_dump()
    data["gates"]["lyapunov"]["lambda_min"] = 0.02
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_custom_ic_out_of_range_rejected() -> None:
    data = load_config(ROOT / "config" / "phase0.yaml").model_dump()
    data["substrate"]["ic_mode"] = "custom"
    data["substrate"]["custom_ic"] = [2.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)
