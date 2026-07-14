"""Tests for the echo-state reservoir."""

from pathlib import Path

import numpy as np
import pytest

from config.schema import load_config
from src.analysis.reservoir_probe import probe_reservoir
from src.memory.reservoir import Reservoir

ROOT = Path(__file__).resolve().parents[1]


def test_spectral_radius_below_one() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml").reservoir
    res = Reservoir(
        dim=config.dim,
        spectral_radius=config.spectral_radius,
        input_scaling=config.input_scaling,
        leak_rate=config.leak_rate,
        tau_r=config.tau_r,
        x_dim=config.x_dim,
        input_dim=config.input_dim,
        density=config.density,
        seed=config.seed,
    )
    assert res.spectral_radius < 1.0
    assert res.spectral_radius <= config.spectral_radius + 1e-6


def test_step_shape() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml").reservoir
    res = Reservoir(
        dim=config.dim,
        spectral_radius=config.spectral_radius,
        input_scaling=config.input_scaling,
        leak_rate=config.leak_rate,
        tau_r=config.tau_r,
        x_dim=config.x_dim,
        input_dim=config.input_dim,
        density=config.density,
        seed=config.seed,
    )
    r = res.step(0.1, np.zeros(config.x_dim), None)
    assert r.shape == (config.dim,)
    assert np.all(np.isfinite(r))


def test_reservoir_bounded() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml").reservoir
    res = Reservoir(
        dim=config.dim,
        spectral_radius=config.spectral_radius,
        input_scaling=config.input_scaling,
        leak_rate=config.leak_rate,
        tau_r=config.tau_r,
        x_dim=config.x_dim,
        input_dim=config.input_dim,
        density=config.density,
        seed=config.seed,
    )
    rng = np.random.default_rng(0)
    for _ in range(500):
        res.step(0.1, rng.normal(0, 0.1, config.x_dim), None)
    assert np.max(np.abs(res.state)) < 10


def test_invalid_spectral_radius_rejected() -> None:
    with pytest.raises(ValueError):
        Reservoir(
            dim=64,
            spectral_radius=1.5,
            input_scaling=0.1,
            leak_rate=0.3,
        )


def test_input_shape_validation() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml").reservoir
    res = Reservoir(
        dim=config.dim,
        spectral_radius=config.spectral_radius,
        input_scaling=config.input_scaling,
        leak_rate=config.leak_rate,
        tau_r=config.tau_r,
        x_dim=config.x_dim,
        input_dim=config.input_dim,
        density=config.density,
        seed=config.seed,
    )
    with pytest.raises(ValueError):
        res.step(0.1, np.zeros(config.x_dim + 1), None)
    with pytest.raises(ValueError):
        res.step(0.1, np.zeros(config.x_dim), np.zeros(config.input_dim + 1))


def test_five_group_semantic_probe_separates_related_signals() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml").reservoir
    result = probe_reservoir(config, sequence_length=60)
    assert result.passed
    assert result.dissimilar_distance > 2.0 * result.similar_distance
