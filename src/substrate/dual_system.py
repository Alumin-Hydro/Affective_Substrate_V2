"""Fast/slow coupling and adaptive damping shared by substrate cores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DualSystem:
    """Slow low-pass dynamics and parasympathetic adaptive damping."""

    eps: NDArray[np.float64]
    gamma_0: float
    kappa: float

    def __post_init__(self) -> None:
        eps = np.asarray(self.eps, dtype=np.float64)
        if eps.ndim != 1 or np.any(eps <= 0):
            raise ValueError("eps must be a positive vector")
        if self.gamma_0 <= 0 or self.kappa < 0:
            raise ValueError("gamma_0 must be positive and kappa non-negative")
        object.__setattr__(self, "eps", eps)

    def slow_derivative(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.eps * (np.abs(x) - y)

    def gamma(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.gamma_0 + self.kappa * np.maximum(y, 0.0)


def rk4_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dt: float,
    derivative,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fourth-order Runge--Kutta step for two coupled vector states."""

    k1x, k1y = derivative(x, y)
    k2x, k2y = derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
    k3x, k3y = derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
    k4x, k4y = derivative(x + dt * k3x, y + dt * k3y)
    return (
        x + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0,
        y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0,
    )
