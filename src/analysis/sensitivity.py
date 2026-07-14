"""Gate B: controlled strong-emotion versus neutral input experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config.schema import AppConfig
from substrate import create_core


@dataclass
class SensitivityResult:
    signal_distance: float
    noise_distance: float
    effect_ratio: float
    passed: bool
    emotional_trajectory: NDArray[np.float64]
    neutral_trajectory: NDArray[np.float64]
    baseline_a_trajectory: NDArray[np.float64]
    baseline_b_trajectory: NDArray[np.float64]
    signal_distance_series: NDArray[np.float64]
    noise_distance_series: NDArray[np.float64]

    def to_dict(self, include_series: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "signal_distance": self.signal_distance,
            "noise_distance": self.noise_distance,
            "effect_ratio": self.effect_ratio,
            "passed": self.passed,
        }
        if include_series:
            result.update(
                {
                    "emotional_trajectory": self.emotional_trajectory.tolist(),
                    "neutral_trajectory": self.neutral_trajectory.tolist(),
                    "signal_distance_series": self.signal_distance_series.tolist(),
                    "noise_distance_series": self.noise_distance_series.tolist(),
                }
            )
        return result


def synthetic_inputs(
    config: AppConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return an emotional sequence and two independent near-neutral controls."""

    gate = config.gates.sensitivity
    steps = gate.steps
    k = config.substrate.k
    time = np.arange(steps, dtype=np.float64) * config.substrate.dt
    direction = np.resize(np.array([1.0, -0.8, 0.7, -0.6]), k)
    carrier = 0.75 + 0.25 * np.sin(0.31 * time)
    envelope = np.ones(steps, dtype=np.float64)
    envelope[: max(1, steps // 10)] = np.linspace(0.0, 1.0, max(1, steps // 10))
    emotional = gate.emotional_amplitude * envelope[:, None] * carrier[:, None] * direction
    emotional += (
        0.2
        * gate.emotional_amplitude
        * np.sin(0.83 * time)[:, None]
        * np.roll(direction, 1)[None, :]
    )
    neutral_a = np.random.default_rng(310).normal(
        0.0, gate.neutral_sigma, (steps, k)
    )
    neutral_b = np.random.default_rng(311).normal(
        0.0, gate.neutral_sigma, (steps, k)
    )
    return emotional, neutral_a, neutral_b


def _run(config: AppConfig, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    core = create_core(config.substrate)
    trajectory = np.empty((inputs.shape[0], config.substrate.k), dtype=np.float64)
    for index, external_input in enumerate(inputs):
        core.step(config.substrate.dt, external_input)
        trajectory[index] = core.state.x
    return trajectory


def run_sensitivity_experiment(config: AppConfig) -> SensitivityResult:
    emotional, neutral_a, neutral_b = synthetic_inputs(config)
    emotional_trajectory = _run(config, emotional)
    neutral_trajectory = _run(config, neutral_a)
    baseline_a = _run(config, neutral_a)
    baseline_b = _run(config, neutral_b)

    signal_series = np.linalg.norm(emotional_trajectory - neutral_trajectory, axis=1)
    noise_series = np.linalg.norm(baseline_a - baseline_b, axis=1)
    evaluation_start = max(1, config.gates.sensitivity.steps // 10)
    signal_distance = float(np.sqrt(np.mean(signal_series[evaluation_start:] ** 2)))
    noise_distance = float(np.sqrt(np.mean(noise_series[evaluation_start:] ** 2)))
    effect_ratio = signal_distance / max(noise_distance, np.finfo(np.float64).tiny)
    passed = bool(
        signal_distance > config.gates.sensitivity.noise_multiple * noise_distance
    )
    return SensitivityResult(
        signal_distance=signal_distance,
        noise_distance=noise_distance,
        effect_ratio=effect_ratio,
        passed=passed,
        emotional_trajectory=emotional_trajectory,
        neutral_trajectory=neutral_trajectory,
        baseline_a_trajectory=baseline_a,
        baseline_b_trajectory=baseline_b,
        signal_distance_series=signal_series,
        noise_distance_series=noise_series,
    )
