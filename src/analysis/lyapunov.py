"""Gate A: maximal Lyapunov exponent using Benettin renormalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config.schema import LyapunovGateConfig
from substrate.base import DynamicalCore


@dataclass
class LyapunovResult:
    lambda_max: float
    bounded: bool
    collapsed: bool
    passed: bool
    max_abs_x: float
    variance: float
    variance_per_dim: NDArray[np.float64]
    ranges: NDArray[np.float64]
    autocorrelation: NDArray[np.float64]
    trajectory: NDArray[np.float64]
    local_exponents: NDArray[np.float64]

    def to_dict(self, include_series: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "lambda_max": self.lambda_max,
            "bounded": self.bounded,
            "collapsed": self.collapsed,
            "passed": self.passed,
            "max_abs_x": self.max_abs_x,
            "variance": self.variance,
            "variance_per_dim": self.variance_per_dim.tolist(),
            "ranges": self.ranges.tolist(),
            "autocorrelation": self.autocorrelation.tolist(),
        }
        if include_series:
            result["trajectory"] = self.trajectory.tolist()
            result["local_exponents"] = self.local_exponents.tolist()
        return result


def normalized_autocorrelation(
    values: NDArray[np.float64], max_lag: int = 200
) -> NDArray[np.float64]:
    """Compute a per-dimension autocorrelation beginning at lag zero."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("values must have shape [time, k] with at least two rows")
    lags = min(max_lag, array.shape[0] - 1)
    result = np.zeros((lags + 1, array.shape[1]), dtype=np.float64)
    for index in range(array.shape[1]):
        centered = array[:, index] - np.mean(array[:, index])
        denominator = float(centered @ centered)
        if denominator <= np.finfo(np.float64).eps:
            result[0, index] = 1.0
            continue
        correlation = np.correlate(centered, centered, mode="full")
        result[:, index] = correlation[array.shape[0] - 1 : array.shape[0] + lags]
        result[:, index] /= denominator
    return result


def estimate_max_lyapunov(
    core: DynamicalCore,
    dt: float,
    gate: LyapunovGateConfig,
    seed: int = 2025,
) -> LyapunovResult:
    """Estimate λ_max in the complete core phase space.

    Gate A is deterministic: callers must provide a core configured with zero
    process noise. The initial perturbation is applied to the entire delay
    history before its phase-space norm is normalized to ``d0``.
    """

    if core.config.noise_sigma != 0.0:  # type: ignore[attr-defined]
        raise ValueError("Gate A requires noise_sigma=0")
    for _ in range(gate.transient):
        core.step(dt, external_input=None)

    reference = core
    perturbed = core.clone()
    rng = np.random.default_rng(seed)
    history_shape = perturbed.state.history.shape
    history_direction = rng.normal(size=history_shape)
    history_direction /= np.linalg.norm(history_direction)
    perturbed.perturb_history(gate.d0 * history_direction)

    ref_phase = reference.phase_vector()
    delta = perturbed.phase_vector() - ref_phase
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm <= np.finfo(np.float64).tiny:
        raise RuntimeError("the selected core did not expose the history perturbation")
    perturbed.set_phase_vector(ref_phase + gate.d0 * delta / delta_norm)

    trajectory: list[NDArray[np.float64]] = []
    local_exponents: list[float] = []
    max_abs = float(np.max(np.abs(reference.state.x)))
    fallback_direction = delta / delta_norm

    for _ in range(gate.intervals):
        for _ in range(gate.renorm_interval):
            reference.step(dt, external_input=None)
            perturbed.step(dt, external_input=None)
            trajectory.append(reference.state.x.copy())
            max_abs = max(max_abs, float(np.max(np.abs(reference.state.x))))

        ref_phase = reference.phase_vector()
        separation = perturbed.phase_vector() - ref_phase
        distance = float(np.linalg.norm(separation))
        if not np.isfinite(distance):
            local_exponents.append(float("inf"))
            break
        if distance <= np.finfo(np.float64).tiny:
            local_exponents.append(
                np.log(np.finfo(np.float64).tiny / gate.d0)
                / (gate.renorm_interval * dt)
            )
            perturbed.set_phase_vector(ref_phase + gate.d0 * fallback_direction)
            continue

        local_exponents.append(
            np.log(distance / gate.d0) / (gate.renorm_interval * dt)
        )
        fallback_direction = separation / distance
        perturbed.set_phase_vector(ref_phase + gate.d0 * fallback_direction)

    trajectory_array = np.asarray(trajectory, dtype=np.float64)
    local_array = np.asarray(local_exponents, dtype=np.float64)
    lambda_max = float(np.mean(local_array)) if local_array.size else float("nan")
    tail = trajectory_array[len(trajectory_array) // 2 :]
    variance_per_dim = np.var(tail, axis=0)
    variance = float(np.mean(variance_per_dim))
    ranges = np.column_stack(
        (np.min(trajectory_array, axis=0), np.max(trajectory_array, axis=0))
    )
    bounded = bool(np.isfinite(max_abs) and max_abs < gate.blowup_threshold)
    collapsed = bool(variance <= gate.collapse_threshold)
    passed = bool(lambda_max > gate.lambda_min and bounded and not collapsed)
    return LyapunovResult(
        lambda_max=lambda_max,
        bounded=bounded,
        collapsed=collapsed,
        passed=passed,
        max_abs_x=max_abs,
        variance=variance,
        variance_per_dim=variance_per_dim,
        ranges=ranges,
        autocorrelation=normalized_autocorrelation(trajectory_array),
        trajectory=trajectory_array,
        local_exponents=local_array,
    )
