"""Headless, reproducible visualizations for Phase 0 experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from numpy.typing import NDArray


def configure_plotting() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [
        next((candidate for candidate in candidates if candidate in available), "DejaVu Sans")
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def _prepare_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_lyapunov_diagnostics(
    trajectory: NDArray[np.float64],
    autocorrelation: NDArray[np.float64],
    dt: float,
    dim_names: list[str],
    path: str | Path,
) -> Path:
    configure_plotting()
    output = _prepare_path(path)
    time = np.arange(trajectory.shape[0]) * dt
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for index, name in enumerate(dim_names):
        axes[0].plot(time, trajectory[:, index], label=name, linewidth=0.8)
        axes[1].plot(
            np.arange(autocorrelation.shape[0]) * dt,
            autocorrelation[:, index],
            label=name,
            linewidth=0.9,
        )
    axes[0].set(title="Autonomous substrate waveform", xlabel="time", ylabel="x")
    axes[1].set(title="Autocorrelation", xlabel="lag", ylabel="correlation")
    axes[0].legend(ncol=2)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_sensitivity(
    emotional: NDArray[np.float64],
    neutral: NDArray[np.float64],
    signal_distance: NDArray[np.float64],
    noise_distance: NDArray[np.float64],
    dt: float,
    path: str | Path,
) -> Path:
    configure_plotting()
    output = _prepare_path(path)
    time = np.arange(emotional.shape[0]) * dt
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(time, emotional[:, 0], label="emotional x[0]")
    axes[0].plot(time, neutral[:, 0], label="neutral x[0]", alpha=0.8)
    axes[0].set(title="Strong-emotion and neutral trajectories", ylabel="x[0]")
    axes[0].legend()
    axes[1].plot(time, signal_distance, label="emotional vs neutral")
    axes[1].plot(time, noise_distance, label="neutral noise baseline")
    axes[1].set(title="Trajectory separation", xlabel="time", ylabel="L2 distance")
    axes[1].legend()
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_bare_substrate(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    gamma: NDArray[np.float64],
    dt: float,
    dim_names: list[str],
    path: str | Path,
) -> Path:
    configure_plotting()
    output = _prepare_path(path)
    time = np.arange(x.shape[0]) * dt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for index, name in enumerate(dim_names):
        axes[0, 0].plot(time, x[:, index], label=name, linewidth=0.8)
        axes[0, 1].plot(time, y[:, index], label=name, linewidth=0.8)
        axes[1, 0].plot(time, gamma[:, index], label=name, linewidth=0.8)
    axes[0, 0].set(title="Fast system x(t)", xlabel="time")
    axes[0, 1].set(title="Slow system y(t)", xlabel="time")
    axes[1, 0].set(title="Adaptive damping gamma_eff(t)", xlabel="time")
    axes[1, 1].plot(x[:, 0], x[:, min(1, x.shape[1] - 1)], linewidth=0.6)
    axes[1, 1].set(title="Fast-system phase portrait", xlabel="x[0]", ylabel="x[1]")
    axes[0, 0].legend(ncol=2)
    figure.savefig(output)
    plt.close(figure)
    return output
