#!/usr/bin/env python3
"""Run Gate B and its reservoir semantic-geometry probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from _common import update_gate_results
from analysis.reservoir_probe import probe_reservoir
from analysis.sensitivity import run_sensitivity_experiment
from analysis.viz import plot_sensitivity
from config.schema import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase0.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_sensitivity_experiment(config)
    reservoir_result = probe_reservoir(config.reservoir)
    figure = plot_sensitivity(
        result.emotional_trajectory,
        result.neutral_trajectory,
        result.signal_distance_series,
        result.noise_distance_series,
        config.substrate.dt,
        ROOT / "artifacts" / "figures" / "gateB_sensitivity.png",
    )
    payload = result.to_dict()
    payload.update(
        {
            "noise_multiple_threshold": config.gates.sensitivity.noise_multiple,
            "reservoir_probe": reservoir_result.to_dict(),
            "figure": str(figure.relative_to(ROOT)),
        }
    )
    artifact = update_gate_results("gate_b", payload)
    status = "PASS" if result.passed else "FAIL"
    reservoir_status = "PASS" if reservoir_result.passed else "FAIL"
    print(
        f"Gate B {status}: signal_dist={result.signal_distance:.6f}, "
        f"noise_dist={result.noise_distance:.6f}, ratio={result.effect_ratio:.2f}"
    )
    print(
        f"Reservoir probe {reservoir_status}: "
        f"semantic separation ratio={reservoir_result.separation_ratio:.2f}"
    )
    print(f"Results: {artifact}")
    return 0 if result.passed and reservoir_result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
