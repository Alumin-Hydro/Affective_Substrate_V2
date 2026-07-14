"""Hindmarsh--Rose bursting core used as the Gate A fallback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config.schema import SubstrateConfig

from .base import DynamicalCore
from .state import SubstrateState


class HindmarshRoseCore(DynamicalCore):
    """Four coupled Hindmarsh--Rose neurons with a hidden recovery state."""

    def __init__(self, config: SubstrateConfig):
        self.config = config.model_copy(deep=True)
        hr = self.config.hindmarsh_rose
        for name in ("a", "b", "c", "d", "r", "s", "x_rest", "input_current"):
            setattr(self, name, np.asarray(getattr(hr, name), dtype=np.float64))
        self.W = np.asarray(hr.W, dtype=np.float64)
        self._rng = np.random.default_rng(self.config.seed)
        self._recovery = np.zeros(self.config.k, dtype=np.float64)
        self._state: SubstrateState
        self.reset()

    @property
    def state(self) -> SubstrateState:
        return self._state

    def reset(self, ic: NDArray[np.float64] | None = None) -> None:
        self._rng = np.random.default_rng(self.config.seed)
        if ic is None and self.config.ic_mode == "custom":
            ic = np.asarray(self.config.custom_ic, dtype=np.float64)
        if ic is None:
            x = self.config.sane_ic_center + self._rng.normal(
                0.0, self.config.sane_ic_jitter, self.config.k
            )
        else:
            x = np.asarray(ic, dtype=np.float64).copy()
            if x.shape != (self.config.k,):
                raise ValueError("ic must have shape [k]")
            if np.max(np.abs(x)) > 1.0:
                raise ValueError("ic must remain in the sane range [-1, 1]")
        self._recovery = self.config.sane_ic_center + self._rng.normal(
            0.0, self.config.sane_ic_jitter, self.config.k
        )
        slow = self.config.sane_ic_center + self._rng.normal(
            0.0, self.config.sane_ic_jitter, self.config.k
        )
        history = np.repeat(x[None, :], self.config.kernel.buffer_len, axis=0)
        self._state = SubstrateState(
            x=x,
            y=slow,
            gamma_eff=1.0 + self.r * np.abs(slow),
            t=0,
            history=history,
        )

    def _derivative(self, x, recovery, slow, input_vector):
        dx = (
            recovery
            - self.a * x**3
            + self.b * x**2
            - slow
            + self.input_current
            + self.W @ x
            + input_vector
        )
        drecovery = self.c - self.d * x**2 - recovery
        dslow = self.r * (self.s * (x - self.x_rest) - slow)
        return dx, drecovery, dslow

    def step(
        self, dt: float, external_input: NDArray[np.float64] | None = None
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if external_input is None:
            input_vector = np.zeros(self.config.k, dtype=np.float64)
        else:
            input_vector = np.asarray(external_input, dtype=np.float64)
            if input_vector.shape != (self.config.k,):
                raise ValueError("external_input must have shape [k]")
        if self.config.noise_sigma:
            input_vector = input_vector + self._rng.normal(
                0.0, self.config.noise_sigma, self.config.k
            )

        x, q, z = self._state.x, self._recovery, self._state.y
        if self.config.integrator == "rk4":
            k1 = self._derivative(x, q, z, input_vector)
            k2 = self._derivative(
                x + 0.5 * dt * k1[0],
                q + 0.5 * dt * k1[1],
                z + 0.5 * dt * k1[2],
                input_vector,
            )
            k3 = self._derivative(
                x + 0.5 * dt * k2[0],
                q + 0.5 * dt * k2[1],
                z + 0.5 * dt * k2[2],
                input_vector,
            )
            k4 = self._derivative(
                x + dt * k3[0],
                q + dt * k3[1],
                z + dt * k3[2],
                input_vector,
            )
            x_new = x + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            q_new = q + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            z_new = z + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        else:
            dx, dq, dz = self._derivative(x, q, z, input_vector)
            x_new, q_new, z_new = x + dt * dx, q + dt * dq, z + dt * dz

        if not all(np.all(np.isfinite(v)) for v in (x_new, q_new, z_new)):
            raise FloatingPointError("Hindmarsh--Rose state became non-finite")
        self._state.history[:-1] = self._state.history[1:]
        self._state.history[-1] = x_new
        self._state.x = x_new
        self._recovery = q_new
        self._state.y = z_new
        self._state.gamma_eff = 1.0 + self.r * np.abs(z_new)
        self._state.t += 1

    def perturb_history(self, delta: NDArray[np.float64]) -> None:
        perturbation = np.asarray(delta, dtype=np.float64)
        if perturbation.shape == (self.config.k,):
            self._state.history += perturbation
        elif perturbation.shape == self._state.history.shape:
            self._state.history += perturbation
        else:
            raise ValueError("delta must have shape [k] or [buffer_len, k]")
        self._state.x = self._state.history[-1].copy()

    def phase_vector(self) -> NDArray[np.float64]:
        return np.concatenate((self._state.x, self._recovery, self._state.y))

    def set_phase_vector(self, vector: NDArray[np.float64]) -> None:
        array = np.asarray(vector, dtype=np.float64)
        expected = 3 * self.config.k
        if array.shape != (expected,):
            raise ValueError(f"phase vector must have shape [{expected}]")
        k = self.config.k
        self._state.x = array[:k].copy()
        self._recovery = array[k : 2 * k].copy()
        self._state.y = array[2 * k :].copy()
        self._state.history[-1] = self._state.x
        self._state.gamma_eff = 1.0 + self.r * np.abs(self._state.y)
