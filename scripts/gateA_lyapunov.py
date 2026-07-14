#!/usr/bin/env python3
"""Run Gate A and persist the maximal Lyapunov result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from _common import update_gate_results
from analysis.lyapunov import estimate_max_lyapunov
from analysis.viz import plot_lyapunov_diagnostics
from config.schema import load_config
from substrate import create_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase0.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.substrate.noise_sigma != 0.0:
        parser.error("Gate A requires substrate.noise_sigma=0")

    result = estimate_max_lyapunov(
        create_core(config.substrate),
        dt=config.substrate.dt,
        gate=config.gates.lyapunov,
    )
    figure = plot_lyapunov_diagnostics(
        result.trajectory,
        result.autocorrelation,
        config.substrate.dt,
        config.substrate.dim_names,
        ROOT / "artifacts" / "figures" / "gateA_lyapunov.png",
    )
    payload = result.to_dict()
    payload.update(
        {
            "core": config.substrate.core,
            "noise_sigma": config.substrate.noise_sigma,
            "integrator": config.substrate.integrator,
            "dt": config.substrate.dt,
            "thresholds": {
                "lambda_min": config.gates.lyapunov.lambda_min,
                "blowup_threshold": config.gates.lyapunov.blowup_threshold,
                "collapse_threshold": config.gates.lyapunov.collapse_threshold,
            },
            "kernel": config.substrate.kernel.model_dump(),
            "figure": str(figure.relative_to(ROOT)),
        }
    )
    artifact = update_gate_results("gate_a", payload)
    status = "PASS" if result.passed else "FAIL"
    print(
        f"Gate A {status}: lambda_max={result.lambda_max:.6f}, "
        f"variance={result.variance:.6f}, max|x|={result.max_abs_x:.6f}"
    )
    print(f"Results: {artifact}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
