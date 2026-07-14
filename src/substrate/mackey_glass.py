"""Mackey--Glass core with replaceable delay kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config.schema import SubstrateConfig

from .base import DynamicalCore
from .dual_system import DualSystem, rk4_pair
from .kernels import dirac_delayed, hybrid_delayed, powerlaw_delayed
from .state import SubstrateState


class MackeyGlassCore(DynamicalCore):
    """Vector Mackey--Glass system coupled to a slow damping state."""

    def __init__(self, config: SubstrateConfig):
        self.config = config.model_copy(deep=True)
        mg = self.config.mackey_glass
        self.beta = np.asarray(mg.beta, dtype=np.float64)
        self.tau_steps = np.asarray(mg.tau_steps, dtype=np.int64)
        self.W = np.asarray(mg.W, dtype=np.float64)
        self.n = mg.n
        self.dual = DualSystem(
            eps=np.asarray(self.config.slow.eps, dtype=np.float64),
            gamma_0=mg.gamma_0,
            kappa=mg.kappa,
        )
        self._state: SubstrateState
        self._rng = np.random.default_rng(self.config.seed)
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

        history = x + self._rng.normal(
            0.0,
            self.config.sane_ic_jitter,
            (self.config.kernel.buffer_len, self.config.k),
        )
        history[-1] = x
        y = np.abs(x) + self._rng.normal(
            0.0, self.config.sane_ic_jitter * 0.25, self.config.k
        )
        y = np.maximum(y, 0.0)
        self._state = SubstrateState(
            x=x,
            y=y,
            gamma_eff=self.dual.gamma(y),
            t=0,
            history=history,
        )

    def _delayed(self) -> NDArray[np.float64]:
        kernel = self.config.kernel
        delayed = np.empty(self.config.k, dtype=np.float64)
        for i, tau in enumerate(self.tau_steps):
            column = self._state.history[:, i : i + 1]
            if kernel.type == "dirac":
                delayed[i] = dirac_delayed(column, int(tau))[0]
            elif kernel.type == "powerlaw":
                delayed[i] = powerlaw_delayed(column, kernel.alpha, kernel.s0)[0]
            else:
                delayed[i] = hybrid_delayed(
                    column, int(tau), kernel.alpha, kernel.s0, kernel.w
                )[0]
        return delayed

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
        delayed = self._delayed()

        def derivative(x, y):
            denominator = 1.0 + np.power(delayed, self.n)
            dx = (
                self.beta * delayed / denominator
                - self.dual.gamma(y) * x
                + self.W @ x
                + input_vector
            )
            return dx, self.dual.slow_derivative(x, y)

        if self.config.integrator == "rk4":
            x_new, y_new = rk4_pair(
                self._state.x, self._state.y, float(dt), derivative
            )
        else:
            dx, dy = derivative(self._state.x, self._state.y)
            x_new = self._state.x + dt * dx
            y_new = self._state.y + dt * dy

        if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(y_new)):
            raise FloatingPointError("Mackey--Glass state became non-finite")
        self._state.history[:-1] = self._state.history[1:]
        self._state.history[-1] = x_new
        self._state.x = x_new
        self._state.y = np.maximum(y_new, 0.0)
        self._state.gamma_eff = self.dual.gamma(self._state.y)
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
        return np.concatenate((self._state.history.ravel(), self._state.y))

    def set_phase_vector(self, vector: NDArray[np.float64]) -> None:
        array = np.asarray(vector, dtype=np.float64)
        history_size = self._state.history.size
        expected = history_size + self.config.k
        if array.shape != (expected,):
            raise ValueError(f"phase vector must have shape [{expected}]")
        self._state.history[:] = array[:history_size].reshape(
            self._state.history.shape
        )
        self._state.x = self._state.history[-1].copy()
        self._state.y = np.maximum(array[history_size:].copy(), 0.0)
        self._state.gamma_eff = self.dual.gamma(self._state.y)
