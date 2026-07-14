#!/usr/bin/env python3
"""Scan hybrid-kernel w and alpha values for a Mackey--Glass chaotic window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.lyapunov import estimate_max_lyapunov
from config.schema import load_config
from substrate import create_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase0.yaml")
    parser.add_argument("--w", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--intervals", type=int, default=80)
    args = parser.parse_args()
    base = load_config(args.config)
    if base.substrate.core != "mackey_glass":
        parser.error("kernel sweep applies only to core=mackey_glass")
    if base.substrate.noise_sigma != 0.0:
        parser.error("kernel sweep requires noise_sigma=0")

    exponent_grid = np.empty((len(args.alpha), len(args.w)), dtype=np.float64)
    records: list[dict[str, float | bool]] = []
    for alpha_index, alpha in enumerate(args.alpha):
        for w_index, w in enumerate(args.w):
            config = base.model_copy(deep=True)
            config.substrate.kernel.w = w
            config.substrate.kernel.alpha = alpha
            config.gates.lyapunov.intervals = args.intervals
            result = estimate_max_lyapunov(
                create_core(config.substrate), config.substrate.dt, config.gates.lyapunov
            )
            exponent_grid[alpha_index, w_index] = result.lambda_max
            records.append(
                {
                    "w": w,
                    "alpha": alpha,
                    "lambda_max": result.lambda_max,
                    "variance": result.variance,
                    "max_abs_x": result.max_abs_x,
                    "passed": result.passed,
                }
            )
            print(
                f"w={w:.3f} alpha={alpha:.3f} lambda={result.lambda_max:.6f} "
                f"{'PASS' if result.passed else 'FAIL'}"
            )

    best = max(records, key=lambda record: float(record["lambda_max"]))
    output = ROOT / "artifacts" / "kernel_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"results": records, "best": best}, handle, indent=2)
        handle.write("\n")

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    image = axis.imshow(exponent_grid, aspect="auto", origin="lower", cmap="coolwarm")
    axis.set_xticks(range(len(args.w)), labels=[f"{value:.2f}" for value in args.w])
    axis.set_yticks(
        range(len(args.alpha)), labels=[f"{value:.2f}" for value in args.alpha]
    )
    axis.set(xlabel="w", ylabel="alpha", title="Maximum Lyapunov exponent")
    figure.colorbar(image, ax=axis, label="lambda_max")
    figure_path = ROOT / "artifacts" / "figures" / "kernel_sweep.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"Best: {best}")
    return 0 if any(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
