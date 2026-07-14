"""Project substrate/reservoir state onto LLM steering vectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from config.schema import InjectionConfig

from src.substrate.state import SubstrateState


@dataclass
class InjectionRecord:
    """Per-step injection diagnostics."""

    layer_norms_raw: dict[int, float] = field(default_factory=dict)
    layer_norms_clipped: dict[int, float] = field(default_factory=dict)
    ratio_to_hidden: dict[int, float] = field(default_factory=dict)
    clip_triggered: dict[int, bool] = field(default_factory=dict)

    def update(
        self,
        layer_idx: int,
        raw: float,
        clipped: float,
        ratio_to_hidden: float,
        clip_triggered: bool,
    ) -> None:
        self.layer_norms_raw[layer_idx] = raw
        self.layer_norms_clipped[layer_idx] = clipped
        self.ratio_to_hidden[layer_idx] = ratio_to_hidden
        self.clip_triggered[layer_idx] = clip_triggered


def _compute_gains(
    value: NDArray[np.float64],
    baseline: NDArray[np.float64],
    scale: NDArray[np.float64],
    amplitude: NDArray[np.float64],
) -> torch.Tensor:
    """Apply a scaled tanh non-linearity to each affective dimension."""

    value = np.asarray(value, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    centered = value - baseline
    tanh_out = np.tanh(scale * centered)
    return torch.from_numpy(amplitude * tanh_out).to(torch.float32)


def compute_injection(
    state: SubstrateState,
    r: NDArray[np.float64],
    V: NDArray[np.float64],
    P: NDArray[np.float64],
    config: InjectionConfig,
    V_slow: NDArray[np.float64] | None = None,
) -> tuple[dict[int, torch.Tensor], InjectionRecord]:
    """Compute per-layer residual-stream deltas and a diagnostic record.

    Parameters
    ----------
    state
        Current substrate fast/slow state.
    r
        Reservoir state, shape ``[N]``.
    V
        Steering directions, shape ``[k, num_layers, d_model]``.
    P
        Reservoir projection, shape ``[d_model, N]``.
    config
        Injection configuration with ``A``, ``B``, ``s``, ``x_baseline``,
        ``y_baseline``, ``cap``, and ``slow_use_separate_directions``.
    V_slow
        Optional separate directions for the slow/parasympathetic term, shape
        ``[k, num_layers, d_model]``. Defaults to ``V`` when not supplied.

    Returns
    -------
    deltas
        Mapping from layer index to delta vector ``[d_model]``.
    record
        ``InjectionRecord`` with per-layer raw/clipped norm information.

    Notes
    -----
    Per-layer delta:

    .. math::

        \\Delta h_L = P r + \\sum_i \\alpha_i V_{i,L} - \\sum_i \\beta_i V'_{i,L}

    where:

    .. math::

        \\alpha_i = A_i \\tanh(s_i (x_i - \\bar{x}_i))

        \\beta_i = B_i \\tanh(s_i (y_i - \\bar{y}_i))
    """

    V = np.asarray(V, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    if V.ndim != 3:
        raise ValueError(f"V must have shape [k, num_layers, d_model]; got {V.ndim}D")
    if P.ndim != 2:
        raise ValueError(f"P must have shape [d_model, N]; got {P.ndim}D")
    if r.ndim != 1:
        raise ValueError(f"r must have shape [N]; got {r.ndim}D")

    k, num_layers, d_model = V.shape
    if P.shape[0] != d_model:
        raise ValueError(
            f"P rows ({P.shape[0]}) must match d_model ({d_model})"
        )
    if P.shape[1] != r.shape[0]:
        raise ValueError(
            f"P columns ({P.shape[1]}) must match r length ({r.shape[0]})"
        )
    if state.x.shape[0] != k or state.y.shape[0] != k:
        raise ValueError(f"state x/y must have length k={k}")

    _check_length("A", config.A, k)
    _check_length("B", config.B, k)
    _check_length("s", config.s, k)
    _check_length("x_baseline", config.x_baseline, k)
    _check_length("y_baseline", config.y_baseline, k)

    alpha = _compute_gains(
        state.x, np.asarray(config.x_baseline), np.asarray(config.s), np.asarray(config.A)
    )
    beta = _compute_gains(
        state.y, np.asarray(config.y_baseline), np.asarray(config.s), np.asarray(config.B)
    )

    V_slow = V if V_slow is None else np.asarray(V_slow, dtype=np.float64)
    if V_slow.shape != V.shape:
        raise ValueError(f"V_slow shape {V_slow.shape} must match V shape {V.shape}")

    r_tensor = torch.from_numpy(r).to(torch.float32)
    P_tensor = torch.from_numpy(P).to(torch.float32)
    V_tensor = torch.from_numpy(V).to(torch.float32)
    V_slow_tensor = torch.from_numpy(V_slow).to(torch.float32)

    reservoir_term = P_tensor @ r_tensor

    deltas: dict[int, torch.Tensor] = {}
    record = InjectionRecord()
    for layer_idx in range(num_layers):
        symp = torch.sum(alpha.unsqueeze(1) * V_tensor[:, layer_idx, :], dim=0)
        parasymp = torch.sum(beta.unsqueeze(1) * V_slow_tensor[:, layer_idx, :], dim=0)
        delta = reservoir_term + symp - parasymp

        raw_norm = float(delta.norm(p=2))
        clipped_norm = min(raw_norm, config.cap)
        clip_triggered = raw_norm > config.cap
        if clip_triggered:
            delta = delta * (clipped_norm / (raw_norm + 1e-12))

        # Ratio-to-hidden is a placeholder until the hidden norm is available.
        ratio = 0.0
        record.update(
            layer_idx=layer_idx,
            raw=raw_norm,
            clipped=clipped_norm,
            ratio_to_hidden=ratio,
            clip_triggered=clip_triggered,
        )
        deltas[layer_idx] = delta

    return deltas, record


def _check_length(name: str, values: list[float], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values; got {len(values)}")


def build_reservoir_projection(
    reservoir_dim: int, d_model: int, seed: int = 17
) -> NDArray[np.float64]:
    """Create a random projection matrix from reservoir to model space."""

    rng = np.random.default_rng(seed)
    P = rng.normal(0.0, 1.0 / np.sqrt(reservoir_dim), (d_model, reservoir_dim))
    return P.astype(np.float64)
