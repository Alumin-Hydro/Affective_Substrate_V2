"""Tests for delay kernels."""

import numpy as np
import pytest

from src.substrate.kernels import (
    dirac_delayed,
    hybrid_delayed,
    powerlaw_delayed,
    powerlaw_weights,
)


@pytest.fixture
def history():
    rng = np.random.default_rng(0)
    return rng.normal(0.5, 0.1, (50, 4))


def test_dirac_delayed_shape(history) -> None:
    delayed = dirac_delayed(history, 10)
    assert delayed.shape == (4,)


def test_powerlaw_weights_normalized() -> None:
    for alpha in [0.3, 0.5, 1.0]:
        for s0 in [1.0, 2.0]:
            weights = powerlaw_weights(50, alpha, s0)
            assert weights.shape == (50,)
            assert np.all(weights >= 0)
            assert np.isclose(weights.sum(), 1.0)


def test_powerlaw_delayed_shape(history) -> None:
    delayed = powerlaw_delayed(history, 0.3, 1.0)
    assert delayed.shape == (4,)


def test_hybrid_delayed_weighting(history) -> None:
    fixed = dirac_delayed(history, 10)
    distributed = powerlaw_delayed(history, 0.3, 1.0)
    mixed = hybrid_delayed(history, 10, 0.3, 1.0, 0.5)
    assert np.allclose(mixed, 0.5 * fixed + 0.5 * distributed)


def test_hybrid_weights_are_bounded(history) -> None:
    mixed = hybrid_delayed(history, 10, 0.3, 1.0, 0.9)
    # With 0.9 weight on dirac, the mixed value is mostly in the convex hull
    assert np.all(np.isfinite(mixed))
    assert mixed.shape == (4,)
