#!/usr/bin/env python3
"""Run the substrate without an LLM and render state diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.viz import plot_bare_substrate
from config.schema import load_config
from substrate import create_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase0.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "figures" / "bare_substrate.png"
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    config = load_config(args.config)
    core = create_core(config.substrate)
    x = np.empty((args.steps, config.substrate.k), dtype=np.float64)
    y = np.empty_like(x)
    gamma = np.empty_like(x)
    for index in range(args.steps):
        core.step(config.substrate.dt)
        x[index] = core.state.x
        y[index] = core.state.y
        gamma[index] = core.state.gamma_eff
    output = plot_bare_substrate(
        x, y, gamma, config.substrate.dt, config.substrate.dim_names, args.output
    )
    print(
        f"Ran {args.steps} autonomous {config.substrate.core} steps; "
        f"max|x|={np.max(np.abs(x)):.6f}; figure={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
