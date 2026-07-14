"""State container shared by all dynamical cores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SubstrateState:
    """Observable fast/slow state and the delay-system history buffer."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gamma_eff: NDArray[np.float64]
    t: int
    history: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.gamma_eff = np.asarray(self.gamma_eff, dtype=np.float64)
        self.history = np.asarray(self.history, dtype=np.float64)
        if self.x.ndim != 1:
            raise ValueError("x must have shape [k]")
        if self.y.shape != self.x.shape or self.gamma_eff.shape != self.x.shape:
            raise ValueError("x, y, and gamma_eff must have the same shape")
        if self.history.ndim != 2 or self.history.shape[1] != self.x.size:
            raise ValueError("history must have shape [buffer_len, k]")
        if not isinstance(self.t, int) or self.t < 0:
            raise ValueError("t must be a non-negative integer")

    def copy(self) -> "SubstrateState":
        """Return a deep numerical copy suitable for diagnostics."""

        return SubstrateState(
            x=self.x.copy(),
            y=self.y.copy(),
            gamma_eff=self.gamma_eff.copy(),
            t=self.t,
            history=self.history.copy(),
        )
