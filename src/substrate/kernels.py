"""Normalized discrete delay kernels."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


def _validate_history(history: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(history, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("history must have shape [buffer_len, k]")
    return array


def dirac_delayed(
    history: NDArray[np.float64], tau_steps: int
) -> NDArray[np.float64]:
    """Return x(t-tau), with history ordered from oldest to newest."""

    array = _validate_history(history)
    if not isinstance(tau_steps, (int, np.integer)) or tau_steps < 0:
        raise ValueError("tau_steps must be a non-negative integer")
    if tau_steps >= array.shape[0]:
        raise ValueError("tau_steps must be smaller than the history buffer")
    return array[-(int(tau_steps) + 1)].copy()


@lru_cache(maxsize=128)
def powerlaw_weights(buffer_len: int, alpha: float, s0: float) -> NDArray[np.float64]:
    """Return newest-first non-negative power-law weights summing to one."""

    if buffer_len < 1:
        raise ValueError("buffer_len must be positive")
    if alpha <= 0 or s0 <= 0:
        raise ValueError("alpha and s0 must be positive")
    lags = np.arange(buffer_len, dtype=np.float64)
    weights = np.power(lags + float(s0), -float(alpha) - 1.0)
    weights /= weights.sum()
    weights.setflags(write=False)
    return weights


def powerlaw_delayed(
    history: NDArray[np.float64], alpha: float, s0: float
) -> NDArray[np.float64]:
    """Convolve all available history with a normalized heavy-tailed kernel."""

    array = _validate_history(history)
    weights = powerlaw_weights(array.shape[0], float(alpha), float(s0))
    return weights @ array[::-1]


def hybrid_delayed(
    history: NDArray[np.float64],
    tau_steps: int,
    alpha: float,
    s0: float,
    w: float,
) -> NDArray[np.float64]:
    """Mix a fixed-delay Dirac component with the power-law component."""

    if not 0.0 <= w <= 1.0:
        raise ValueError("w must lie in [0, 1]")
    fixed = dirac_delayed(history, tau_steps)
    distributed = powerlaw_delayed(history, alpha, s0)
    return float(w) * fixed + (1.0 - float(w)) * distributed
