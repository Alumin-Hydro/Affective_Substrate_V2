"""Autonomous dynamical substrate cores."""

from __future__ import annotations

from config.schema import SubstrateConfig

from .base import DynamicalCore
from .hindmarsh_rose import HindmarshRoseCore
from .mackey_glass import MackeyGlassCore


def create_core(config: SubstrateConfig) -> DynamicalCore:
    """Instantiate the core selected by configuration."""

    if config.core == "mackey_glass":
        return MackeyGlassCore(config)
    if config.core == "hindmarsh_rose":
        return HindmarshRoseCore(config)
    raise ValueError(f"unsupported substrate core: {config.core}")


__all__ = [
    "DynamicalCore",
    "HindmarshRoseCore",
    "MackeyGlassCore",
    "create_core",
]
