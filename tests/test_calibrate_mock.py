"""Tests for the mock calibration path and robust injection hooks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn

from scripts.calibrate_injection import MockLM
from src.llm.hooks import InjectionHook


REPO_ROOT = Path(__file__).resolve().parent.parent


class TinyLM(nn.Module):
    """Small decoder-shaped model for hook tests; no model download required."""

    def __init__(self, d_model: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(num_layers)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class TupleLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def test_injection_hook_modifies_tiny_model_and_can_be_removed() -> None:
    model = TinyLM()
    hidden = torch.zeros(1, 3, 8)
    delta = torch.arange(8, dtype=torch.float32)
    hook = InjectionHook(model, inject_layers=[1])

    assert hook.attached is True
    hook.set_delta(1, delta)
    output = model(hidden)
    assert torch.allclose(output, delta.view(1, 1, -1).expand_as(hidden))
    assert hook.last_record is not None

    hook.clear()
    assert torch.equal(model(hidden), hidden)
    assert hook.last_record is None

    hook.set_delta(1, delta)
    hook.remove()
    hook.remove()  # remove is intentionally idempotent
    assert hook.attached is False
    assert torch.equal(model(hidden), hidden)


def test_injection_hook_preserves_tuple_layer_outputs() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList([TupleLayer()])
    hook = InjectionHook(model, inject_layers=[0])
    hook.set_delta(0, torch.ones(8))

    hidden, cache = model.layers[0](torch.zeros(1, 2, 8))

    assert torch.equal(hidden, torch.ones(1, 2, 8))
    assert cache == "cache"
    hook.remove()


def test_injection_hook_is_noop_for_mock_without_layers() -> None:
    model = MockLM(d_model=8, device=torch.device("cpu"))
    hook = InjectionHook(model, inject_layers=[15, 18])

    assert hook.attached is False
    hook.set_deltas({15: torch.ones(8), 18: torch.zeros(8)})
    hook.clear()
    hook.remove()


def test_calibrate_mock_main_flow_without_direction_artifact(tmp_path: Path) -> None:
    missing_directions = tmp_path / "directions.pt"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate_injection.py"),
            "--device",
            "mock",
            "--dtype",
            "float32",
            "--directions",
            str(missing_directions),
            "--A_sweep",
            "0.01,0.02",
            "--max_new_tokens",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "using deterministic mock directions" in result.stdout
    summary_path = tmp_path / "calibration_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [entry["A"] for entry in summary] == [0.01, 0.02]
    assert set(summary[0]["per_layer_95th"]) == {"15", "18", "21", "24"}


def test_extract_reuses_complete_outputs_before_model_loading(tmp_path: Path) -> None:
    (tmp_path / "directions.pt").write_bytes(b"existing")
    (tmp_path / "directions_meta.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "extract_directions.py"),
            "--config",
            str(tmp_path / "missing-config.yaml"),
            "--output_dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Reusing existing extraction artifacts" in result.stdout
