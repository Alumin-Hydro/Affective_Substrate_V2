"""Tests for substrate cores."""

from pathlib import Path

import numpy as np
import pytest

from config.schema import load_config
from src.substrate import create_core
from src.substrate.hindmarsh_rose import HindmarshRoseCore
from src.substrate.mackey_glass import MackeyGlassCore

ROOT = Path(__file__).resolve().parents[1]


def test_mackey_glass_step_and_boundedness() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    data = config.model_dump()
    data["substrate"]["core"] = "mackey_glass"
    data["substrate"]["noise_sigma"] = 0.0
    cfg = config.model_validate(data).substrate

    core = MackeyGlassCore(cfg)
    for _ in range(500):
        core.step(cfg.dt)
    assert np.all(np.isfinite(core.state.x))
    assert np.max(np.abs(core.state.x)) < 50
    assert core.state.history.shape == (cfg.kernel.buffer_len, cfg.k)


def test_hindmarsh_rose_step_and_boundedness() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    cfg = config.substrate
    core = HindmarshRoseCore(cfg)
    for _ in range(500):
        core.step(cfg.dt)
    assert np.all(np.isfinite(core.state.x))
    assert np.max(np.abs(core.state.x)) < 50


def test_reset_uses_sane_ic() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    cfg = config.substrate
    core = create_core(cfg)
    core.reset()
    assert np.max(np.abs(core.state.x)) < 1.0
    assert np.max(np.abs(core.state.y)) < 1.0


def test_history_rolls() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    cfg = config.substrate
    core = create_core(cfg)
    core.reset()
    old_history = core.state.history.copy()
    core.step(cfg.dt)
    assert not np.array_equal(core.state.history, old_history)
    assert np.array_equal(core.state.history[-1], core.state.x)


def test_external_input_shape() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    cfg = config.substrate
    core = create_core(cfg)
    core.step(cfg.dt, np.zeros(cfg.k))
    with pytest.raises(ValueError):
        core.step(cfg.dt, np.zeros(cfg.k + 1))


def test_perturb_history() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    cfg = config.substrate
    core = create_core(cfg)
    core.reset()
    delta = np.full(cfg.k, 1e-8)
    core.perturb_history(delta)
    assert np.allclose(core.state.x, core.state.history[-1])


def test_phase_vector_roundtrip() -> None:
    config = load_config(ROOT / "config" / "phase0.yaml")
    cfg = config.substrate
    core = create_core(cfg)
    core.reset()
    vector = core.phase_vector()
    core.set_phase_vector(vector)
    assert np.allclose(core.phase_vector(), vector)
