"""Abstract contract for replaceable dynamical substrate cores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .state import SubstrateState


class DynamicalCore(ABC):
    """Interface required by gates and higher-level substrate consumers."""

    @abstractmethod
    def step(
        self, dt: float, external_input: NDArray[np.float64] | None = None
    ) -> None:
        """Advance one integration step and roll the history buffer."""

    @abstractmethod
    def reset(self, ic: NDArray[np.float64] | None = None) -> None:
        """Reset to a sane initial condition unless an explicit IC is supplied."""

    @property
    @abstractmethod
    def state(self) -> SubstrateState:
        """Return the mutable current state."""

    @abstractmethod
    def perturb_history(self, delta: NDArray[np.float64]) -> None:
        """Apply a perturbation to the delay-system history phase space."""

    @abstractmethod
    def phase_vector(self) -> NDArray[np.float64]:
        """Flatten the independent phase variables used by Gate A."""

    @abstractmethod
    def set_phase_vector(self, vector: NDArray[np.float64]) -> None:
        """Restore a flattened phase vector for Benettin renormalization."""

    def clone(self) -> "DynamicalCore":
        """Create an independent, bit-for-bit copy of the core."""

        return deepcopy(self)
