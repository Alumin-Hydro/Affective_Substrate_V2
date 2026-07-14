"""Tests for the LLM injection projection layer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config.schema import InjectionConfig
from src.llm.projection import compute_injection
from src.substrate.state import SubstrateState


def make_state(k: int = 4) -> SubstrateState:
    return SubstrateState(
        x=np.ones(k, dtype=np.float64) * 0.6,
        y=np.ones(k, dtype=np.float64) * 0.5,
        gamma_eff=np.ones(k, dtype=np.float64) * 0.1,
        t=0,
        history=np.ones((10, k), dtype=np.float64) * 0.5,
    )


def make_config(k: int = 4, cap: float = 8.0) -> InjectionConfig:
    return InjectionConfig(
        A=[0.1] * k,
        B=[0.05] * k,
        s=[1.0] * k,
        x_baseline=[0.5] * k,
        y_baseline=[0.5] * k,
        cap=cap,
        slow_use_separate_directions=False,
    )


def test_dimension_contracts() -> None:
    k, num_layers, d_model, N = 4, 3, 16, 8
    state = make_state(k)
    r = np.ones(N, dtype=np.float64)
    V = np.ones((k, num_layers, d_model), dtype=np.float64) / np.sqrt(d_model)
    P = np.ones((d_model, N), dtype=np.float64) / np.sqrt(N)
    config = make_config(k)

    deltas, record = compute_injection(state, r, V, P, config)

    assert sorted(deltas.keys()) == list(range(num_layers))
    for layer_idx, delta in deltas.items():
        assert isinstance(delta, torch.Tensor)
        assert delta.shape == (d_model,), f"layer {layer_idx} delta shape mismatch"
    assert record.layer_norms_raw.keys() == deltas.keys()
    assert record.layer_norms_clipped.keys() == deltas.keys()
    assert record.ratio_to_hidden.keys() == deltas.keys()
    assert record.clip_triggered.keys() == deltas.keys()


def test_injection_record_contents() -> None:
    k, num_layers, d_model, N = 4, 2, 8, 4
    state = make_state(k)
    r = np.zeros(N, dtype=np.float64)
    V = np.ones((k, num_layers, d_model), dtype=np.float64) / np.sqrt(d_model)
    P = np.zeros((d_model, N), dtype=np.float64)
    config = make_config(k, cap=0.5)

    deltas, record = compute_injection(state, r, V, P, config)

    for layer_idx in range(num_layers):
        raw = record.layer_norms_raw[layer_idx]
        clipped = record.layer_norms_clipped[layer_idx]
        assert raw >= 0.0
        assert clipped <= raw
        assert clipped <= config.cap + 1e-8
        if raw > config.cap:
            assert record.clip_triggered[layer_idx] is True
        else:
            assert record.clip_triggered[layer_idx] is False


def test_invalid_dimensions() -> None:
    state = make_state(k=4)
    config = make_config(k=4)
    r = np.ones(8, dtype=np.float64)
    V = np.ones((4, 3, 16), dtype=np.float64)

    P_bad_rows = np.ones((8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="P rows"):
        compute_injection(state, r, V, P_bad_rows, config)

    P_bad_cols = np.ones((16, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="P columns"):
        compute_injection(state, r, V, P_bad_cols, config)

    V_bad = np.ones((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="V must have shape"):
        compute_injection(state, r, V_bad, np.ones((16, 8)), config)

    r_bad = np.ones((8, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="r must have shape"):
        compute_injection(state, r_bad, V, np.ones((16, 8)), config)

    config_bad = make_config(k=3)
    with pytest.raises(ValueError, match="A must contain"):
        compute_injection(state, r, V, np.ones((16, 8)), config_bad)


def test_slow_separate_directions() -> None:
    k, num_layers, d_model, N = 4, 2, 8, 4
    state = make_state(k)
    state.x = np.ones(k) * 0.8
    state.y = np.ones(k) * 0.2
    r = np.zeros(N, dtype=np.float64)
    V = np.ones((k, num_layers, d_model), dtype=np.float64) / np.sqrt(d_model)
    V_slow = -np.ones((k, num_layers, d_model), dtype=np.float64) / np.sqrt(d_model)
    P = np.zeros((d_model, N), dtype=np.float64)
    config = make_config(k)

    deltas_default, _ = compute_injection(state, r, V, P, config)
    deltas_separate, _ = compute_injection(state, r, V, P, config, V_slow=V_slow)

    for layer_idx in range(num_layers):
        # With V_slow = -V, the total delta is P@r + alpha*V - beta*(-V) = (alpha + beta)*V
        # Default is alpha*V - beta*V = (alpha - beta)*V
        # So norm(separate) / norm(default) = (alpha + beta) / (alpha - beta) > 0
        # Here alpha>0 and beta<0, so the magnitudes should differ predictably.
        assert not torch.isclose(
            deltas_default[layer_idx], deltas_separate[layer_idx], atol=1e-6
        ).all()
        assert deltas_separate[layer_idx].norm(p=2) > 0.0
        assert deltas_default[layer_idx].norm(p=2) > 0.0


def test_clip_reduces_norm() -> None:
    k, num_layers, d_model, N = 4, 2, 8, 4
    state = make_state(k)
    state.x = np.ones(k) * 10.0  # drive alpha near A
    r = np.zeros(N, dtype=np.float64)
    V = np.ones((k, num_layers, d_model), dtype=np.float64) / np.sqrt(d_model)
    P = np.zeros((d_model, N), dtype=np.float64)
    config = make_config(k, cap=0.1)

    deltas, record = compute_injection(state, r, V, P, config)
    for layer_idx in range(num_layers):
        assert record.clip_triggered[layer_idx] is True
        assert record.layer_norms_clipped[layer_idx] <= config.cap + 1e-8
        assert torch.isclose(deltas[layer_idx].norm(p=2), torch.tensor(config.cap), atol=1e-5)
