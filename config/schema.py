"""Pydantic v2 models for the project configuration."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MackeyGlassConfig(StrictModel):
    beta: list[float]
    gamma_0: float = Field(gt=0)
    kappa: float = Field(ge=0)
    n: int = Field(ge=2)
    tau_steps: list[int]
    W: list[list[float]]


class HindmarshRoseConfig(StrictModel):
    a: list[float]
    b: list[float]
    c: list[float]
    d: list[float]
    r: list[float]
    s: list[float]
    x_rest: list[float]
    input_current: list[float]
    W: list[list[float]]


class SlowConfig(StrictModel):
    eps: list[float]


class KernelConfig(StrictModel):
    type: Literal["dirac", "powerlaw", "hybrid"] = "hybrid"
    w: float = Field(default=0.7, ge=0, le=1)
    alpha: float = Field(default=0.3, gt=0)
    s0: float = Field(default=1.0, gt=0)
    buffer_len: int = Field(default=500, ge=2)


class SubstrateConfig(StrictModel):
    core: Literal["mackey_glass", "hindmarsh_rose"] = "mackey_glass"
    k: int = Field(default=4, gt=0)
    dim_names: list[str]
    integrator: Literal["rk4", "euler"] = "rk4"
    dt: float = Field(default=0.1, gt=0)
    noise_sigma: float = Field(default=0.0, ge=0)
    seed: int = 7
    ic_mode: Literal["sane", "custom"] = "sane"
    sane_ic_center: float = Field(default=0.5, ge=-1.0, le=1.0)
    sane_ic_jitter: float = Field(default=0.01, ge=0, le=0.1)
    custom_ic: list[float] | None = None
    mackey_glass: MackeyGlassConfig
    hindmarsh_rose: HindmarshRoseConfig
    slow: SlowConfig
    kernel: KernelConfig

    @model_validator(mode="after")
    def validate_dimensions(self) -> "SubstrateConfig":
        if len(self.dim_names) != self.k or len(set(self.dim_names)) != self.k:
            raise ValueError("dim_names must contain k unique names")

        mg = self.mackey_glass
        for name, values in {
            "mackey_glass.beta": mg.beta,
            "mackey_glass.tau_steps": mg.tau_steps,
            "slow.eps": self.slow.eps,
        }.items():
            if len(values) != self.k:
                raise ValueError(f"{name} must contain k values")
        if any(tau < 1 or tau >= self.kernel.buffer_len for tau in mg.tau_steps):
            raise ValueError("every tau_steps value must be in [1, buffer_len)")
        if any(eps <= 0 for eps in self.slow.eps):
            raise ValueError("slow.eps values must be positive")
        if mg.n % 2:
            raise ValueError("mackey_glass.n must be even for signed states")
        self._validate_matrix("mackey_glass.W", mg.W)

        hr = self.hindmarsh_rose
        for name in ("a", "b", "c", "d", "r", "s", "x_rest", "input_current"):
            if len(getattr(hr, name)) != self.k:
                raise ValueError(f"hindmarsh_rose.{name} must contain k values")
        self._validate_matrix("hindmarsh_rose.W", hr.W)

        if self.ic_mode == "custom":
            if self.custom_ic is None or len(self.custom_ic) != self.k:
                raise ValueError("custom_ic must contain k values when ic_mode=custom")
            if max(abs(value) for value in self.custom_ic) > 1.0:
                raise ValueError("custom_ic must remain in the sane range [-1, 1]")
        elif self.custom_ic is not None:
            raise ValueError("custom_ic is only valid when ic_mode=custom")
        return self

    def _validate_matrix(self, name: str, matrix: list[list[float]]) -> None:
        if len(matrix) != self.k or any(len(row) != self.k for row in matrix):
            raise ValueError(f"{name} must have shape [k, k]")


class ReservoirConfig(StrictModel):
    dim: int = Field(default=128, gt=0)
    spectral_radius: float = Field(default=0.95, gt=0, lt=1)
    leak_rate: float = Field(default=0.3, gt=0, le=1)
    tau_r: float = Field(default=5.0, gt=0)
    input_scaling: float = Field(default=0.1, gt=0)
    density: float = Field(default=0.1, gt=0, le=1)
    x_dim: int = Field(default=4, gt=0)
    input_dim: int = Field(default=4, gt=0)
    seed: int = 17


class InjectionConfig(StrictModel):
    A: list[float]
    B: list[float]
    s: list[float]
    x_baseline: list[float]
    y_baseline: list[float]
    cap: float = Field(default=8.0, gt=0)
    slow_use_separate_directions: bool = False


class LLMConfig(StrictModel):
    model_name: str
    dtype: str = "bfloat16"
    inject_layers: list[int]
    deterministic: bool = True
    injection: InjectionConfig


class FeedbackConfig(StrictModel):
    enabled: bool = True
    classifier: Literal["vad", "lexicon"] = "vad"
    steps_per_turn: int = Field(default=50, gt=0)


class LyapunovGateConfig(StrictModel):
    lambda_min: float = 0.01
    blowup_threshold: float = Field(default=100.0, gt=0)
    collapse_threshold: float = Field(default=0.001, gt=0)
    transient: int = Field(default=2000, ge=0)
    renorm_interval: int = Field(default=20, gt=0)
    intervals: int = Field(default=200, gt=1)
    d0: float = Field(default=1e-8, gt=0)

    @model_validator(mode="after")
    def keep_gate_a_thresholds_fixed(self) -> "LyapunovGateConfig":
        fixed = {
            "lambda_min": (self.lambda_min, 0.01),
            "blowup_threshold": (self.blowup_threshold, 100.0),
            "collapse_threshold": (self.collapse_threshold, 0.001),
        }
        for name, (actual, expected) in fixed.items():
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"{name} is fixed at {expected} by the Phase 0 gate")
        return self


class SensitivityGateConfig(StrictModel):
    noise_multiple: float = Field(default=5.0, gt=0)
    steps: int = Field(default=800, gt=10)
    emotional_amplitude: float = Field(default=0.25, gt=0)
    neutral_sigma: float = Field(default=0.0005, gt=0)

    @model_validator(mode="after")
    def keep_gate_b_threshold_fixed(self) -> "SensitivityGateConfig":
        if not math.isclose(self.noise_multiple, 5.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("noise_multiple is fixed at 5.0 by the Phase 0 gate")
        return self


class GatesConfig(StrictModel):
    lyapunov: LyapunovGateConfig
    sensitivity: SensitivityGateConfig


class AppConfig(StrictModel):
    substrate: SubstrateConfig
    reservoir: ReservoirConfig
    llm: LLMConfig
    feedback: FeedbackConfig
    gates: GatesConfig


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML file and validate every field with Pydantic v2."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return AppConfig.model_validate(data)
