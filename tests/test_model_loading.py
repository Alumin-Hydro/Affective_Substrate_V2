"""Tests for offline HuggingFace loading helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from urllib.error import URLError

import pytest
import torch

import src.llm.model_loading as model_loading


def test_configure_hf_environment_sets_proxy_and_mirror(monkeypatch) -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "HF_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)

    endpoint = model_loading.configure_hf_environment(
        proxy="http://127.0.0.1:7890", hf_mirror="https://mirror.example/"
    )

    assert endpoint == "https://mirror.example"
    assert model_loading.os.environ["HF_ENDPOINT"] == "https://mirror.example"
    assert model_loading.os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_resolve_model_source_accepts_local_snapshot(tmp_path: Path) -> None:
    source, is_local = model_loading.resolve_model_source("remote/model", tmp_path)

    assert source == str(tmp_path.resolve())
    assert is_local is True


def test_resolve_model_source_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="downloaded model/snapshot directory"):
        model_loading.resolve_model_source("remote/model", tmp_path / "missing")


def test_probe_hf_endpoint_reports_network_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise URLError("connection timed out")

    monkeypatch.setattr(model_loading, "urlopen", fail)
    status = model_loading.probe_hf_endpoint("https://huggingface.invalid", timeout=0.1)

    assert status.reachable is False
    assert status.endpoint == "https://huggingface.invalid"
    assert "timed out" in status.detail


def test_remote_load_error_lists_all_fallbacks() -> None:
    message = model_loading._load_error_message(
        source="org/model",
        is_local=False,
        status=model_loading.EndpointStatus(
            "https://huggingface.co", False, "connection timed out"
        ),
        error=OSError("download failed"),
    )

    assert "--local-model-dir" in message
    assert "--hf-mirror" in message
    assert "HF_ENDPOINT" in message
    assert "HTTPS_PROXY" in message


def test_unreachable_endpoint_uses_existing_hf_cache_only(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeTokenizerLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(kwargs)
            return object()

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(kwargs)
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerLoader
    fake_transformers.AutoModelForCausalLM = FakeModelLoader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        model_loading,
        "probe_hf_endpoint",
        lambda endpoint, timeout: model_loading.EndpointStatus(
            endpoint, False, "connection timed out"
        ),
    )

    model_loading.load_causal_lm(
        model_name="org/cached-model", device="cpu", dtype=torch.float32
    )

    assert len(calls) == 2
    assert all(call["local_files_only"] is True for call in calls)
