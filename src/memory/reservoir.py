"""Sparse echo-state reservoir used as L1 working memory."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Reservoir:
    """Continuous-time leaky echo-state network with deterministic weights."""

    def __init__(
        self,
        dim: int,
        spectral_radius: float,
        input_scaling: float,
        leak_rate: float,
        tau_r: float = 5.0,
        x_dim: int = 4,
        input_dim: int = 4,
        density: float = 0.1,
        seed: int = 17,
    ) -> None:
        if dim <= 0 or x_dim <= 0 or input_dim <= 0:
            raise ValueError("dim, x_dim, and input_dim must be positive")
        if not 0.0 < spectral_radius < 1.0:
            raise ValueError("spectral_radius must lie in (0, 1)")
        if input_scaling <= 0 or tau_r <= 0:
            raise ValueError("input_scaling and tau_r must be positive")
        if not 0.0 < leak_rate <= 1.0 or not 0.0 < density <= 1.0:
            raise ValueError("leak_rate and density must lie in (0, 1]")

        self.dim = int(dim)
        self.x_dim = int(x_dim)
        self.input_dim = int(input_dim)
        self.leak_rate = float(leak_rate)
        self.tau_r = float(tau_r)
        self._rng = np.random.default_rng(seed)

        weights = self._rng.normal(0.0, 1.0, (self.dim, self.dim))
        mask = self._rng.random((self.dim, self.dim)) < density
        weights *= mask
        radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
        if radius <= np.finfo(np.float64).eps:
            raise ValueError("random reservoir has zero spectral radius")
        self.W_r = weights * (float(spectral_radius) / radius)
        self.W_x = self._rng.normal(
            0.0, input_scaling / np.sqrt(self.x_dim), (self.dim, self.x_dim)
        )
        self.W_in = self._rng.normal(
            0.0,
            input_scaling / np.sqrt(self.input_dim),
            (self.dim, self.input_dim),
        )
        self._state = np.zeros(self.dim, dtype=np.float64)

    @property
    def state(self) -> NDArray[np.float64]:
        return self._state

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.W_r))))

    def reset(self, state: NDArray[np.float64] | None = None) -> None:
        if state is None:
            self._state = np.zeros(self.dim, dtype=np.float64)
            return
        array = np.asarray(state, dtype=np.float64)
        if array.shape != (self.dim,):
            raise ValueError("reservoir state must have shape [dim]")
        self._state = array.copy()

    def step(
        self,
        dt: float,
        x: NDArray[np.float64],
        u: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Advance dr/dt=-r/tau_r+tanh(W_r r+W_x x+W_in u)."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.shape != (self.x_dim,):
            raise ValueError("x must have shape [x_dim]")
        if u is None:
            u_array = np.zeros(self.input_dim, dtype=np.float64)
        else:
            u_array = np.asarray(u, dtype=np.float64)
            if u_array.shape != (self.input_dim,):
                raise ValueError("u must have shape [input_dim]")
        activation = np.tanh(
            self.W_r @ self._state + self.W_x @ x_array + self.W_in @ u_array
        )
        derivative = -self._state / self.tau_r + activation
        self._state = self._state + self.leak_rate * float(dt) * derivative
        if not np.all(np.isfinite(self._state)):
            raise FloatingPointError("reservoir state became non-finite")
        return self._state
