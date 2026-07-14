"""Top-level exports for the LLM injection pipeline."""

from __future__ import annotations

from .directions import (
    DIMS,
    DEFAULT_CONTRAST_PAIRS,
    extract_directions,
    get_default_contrast_pairs,
    load_directions,
    load_directions_meta,
    orthogonalize_directions,
    save_directions,
)
from .hooks import InjectionHook, InjectionRecord
from .projection import build_reservoir_projection, compute_injection

__all__ = [
    "DIMS",
    "DEFAULT_CONTRAST_PAIRS",
    "InjectionHook",
    "InjectionRecord",
    "build_reservoir_projection",
    "compute_injection",
    "extract_directions",
    "get_default_contrast_pairs",
    "load_directions",
    "load_directions_meta",
    "orthogonalize_directions",
    "save_directions",
]
