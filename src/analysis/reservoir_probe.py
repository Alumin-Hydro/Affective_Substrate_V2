"""Synthetic semantic-geometry probe for the L1 reservoir."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config.schema import ReservoirConfig
from memory.reservoir import Reservoir


@dataclass
class ReservoirProbeResult:
    similar_distance: float
    dissimilar_distance: float
    separation_ratio: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "similar_distance": self.similar_distance,
            "dissimilar_distance": self.dissimilar_distance,
            "separation_ratio": self.separation_ratio,
            "passed": self.passed,
        }


def _make_reservoir(config: ReservoirConfig) -> Reservoir:
    return Reservoir(
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


def _cosine_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(1.0 - np.dot(a, b) / denominator)


def probe_reservoir(
    config: ReservoirConfig, sequence_length: int = 80, seed: int = 404
) -> ReservoirProbeResult:
    """Check that five related signal pairs remain closer than unrelated pairs."""

    if config.x_dim != config.input_dim:
        raise ValueError("the Phase 0 probe requires x_dim == input_dim")
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(5, config.input_dim))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    states_a: list[NDArray[np.float64]] = []
    states_b: list[NDArray[np.float64]] = []

    time = np.linspace(0.0, 4.0 * np.pi, sequence_length)
    modulation = 0.8 + 0.2 * np.sin(time)
    for prototype in prototypes:
        pair_states = []
        for noise_seed in (rng.integers(0, 2**32), rng.integers(0, 2**32)):
            local_rng = np.random.default_rng(int(noise_seed))
            sequence = modulation[:, None] * prototype[None, :]
            sequence += local_rng.normal(0.0, 0.015, sequence.shape)
            reservoir = _make_reservoir(config)
            for signal in sequence:
                reservoir.step(0.5, x=0.25 * signal, u=signal)
            pair_states.append(reservoir.state.copy())
        states_a.append(pair_states[0])
        states_b.append(pair_states[1])

    similar = np.array(
        [_cosine_distance(a, b) for a, b in zip(states_a, states_b)],
        dtype=np.float64,
    )
    dissimilar = np.array(
        [
            _cosine_distance(states_a[index], states_b[(index + 1) % len(states_b)])
            for index in range(len(states_a))
        ],
        dtype=np.float64,
    )
    similar_mean = float(np.mean(similar))
    dissimilar_mean = float(np.mean(dissimilar))
    ratio = dissimilar_mean / max(similar_mean, np.finfo(np.float64).tiny)
    return ReservoirProbeResult(
        similar_distance=similar_mean,
        dissimilar_distance=dissimilar_mean,
        separation_ratio=ratio,
        passed=bool(dissimilar_mean > 2.0 * similar_mean),
    )
