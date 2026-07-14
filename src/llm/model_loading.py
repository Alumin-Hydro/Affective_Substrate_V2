"""Shared HuggingFace model loading with offline and mirror fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch


DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_MIRROR = "https://hf-mirror.com"


@dataclass(frozen=True)
class EndpointStatus:
    """Result of a lightweight HuggingFace endpoint reachability check."""

    endpoint: str
    reachable: bool
    detail: str


def configure_hf_environment(
    proxy: str | None = None, hf_mirror: str | None = None
) -> str:
    """Configure download environment variables before importing Transformers."""

    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = proxy
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = hf_mirror.rstrip("/")
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def probe_hf_endpoint(endpoint: str, timeout: float = 5.0) -> EndpointStatus:
    """Check whether an HF endpoint is reachable without downloading model data."""

    request = Request(
        endpoint,
        headers={"User-Agent": "Affective-Substrate-network-check/1.0"},
        method="HEAD",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", "connected")
        return EndpointStatus(endpoint, True, f"HTTP {status}")
    except HTTPError as exc:
        # Any HTTP response proves DNS, TCP, TLS, and the endpoint are reachable.
        return EndpointStatus(endpoint, True, f"HTTP {exc.code}")
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return EndpointStatus(endpoint, False, str(reason))


def resolve_model_source(
    model_name: str, local_model_dir: str | Path | None
) -> tuple[str, bool]:
    """Resolve a model identifier or validate a fully downloaded local snapshot."""

    if local_model_dir is None:
        return model_name, False

    path = Path(local_model_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local model directory does not exist: {path}. Point --local-model-dir "
            "to a downloaded model/snapshot directory containing config.json."
        )
    return str(path.resolve()), True


def _load_error_message(
    *,
    source: str,
    is_local: bool,
    status: EndpointStatus | None,
    error: Exception,
) -> str:
    original = f"{type(error).__name__}: {error}"
    if is_local:
        return (
            f"Failed to load the local model from {source}. Verify that this is a complete "
            "HuggingFace model/snapshot directory containing config, tokenizer, and weight "
            f"files. Original error: {original}"
        )

    network = "Network probe was not run."
    if status is not None:
        network = (
            f"Network probe reached {status.endpoint} ({status.detail})."
            if status.reachable
            else f"Network probe could not reach {status.endpoint}: {status.detail}."
        )
    return (
        f"Failed to load HuggingFace model {source}. {network}\n"
        "Available offline/network fallbacks:\n"
        "  1. Pass --local-model-dir PATH to a complete downloaded model snapshot.\n"
        f"  2. Pass --hf-mirror (defaults to {DEFAULT_HF_MIRROR}) or set HF_ENDPOINT.\n"
        "  3. Set HTTPS_PROXY=http://127.0.0.1:7890 (or pass --proxy with that URL).\n"
        f"Original error: {original}"
    )


def load_causal_lm(
    *,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    local_model_dir: str | Path | None = None,
    hf_mirror: str | None = None,
    proxy: str | None = None,
    network_timeout: float = 5.0,
) -> tuple[Any, Any]:
    """Load a causal LM/tokenizer from a local snapshot or HuggingFace endpoint."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this Mac")

    endpoint = configure_hf_environment(proxy=proxy, hf_mirror=hf_mirror)
    source, is_local = resolve_model_source(model_name, local_model_dir)

    status = None
    if is_local:
        print(f"Using local model directory: {source}")
    else:
        offline_requested = any(
            os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        status = (
            EndpointStatus(endpoint, False, "offline mode enabled by environment")
            if offline_requested
            else probe_hf_endpoint(endpoint, timeout=network_timeout)
        )
        if status.reachable:
            print(f"HuggingFace endpoint reachable: {endpoint} ({status.detail})")
        else:
            print(
                f"Warning: HuggingFace endpoint is not reachable: {endpoint} "
                f"({status.detail}). Trying the local HF cache before failing."
            )

    # Import lazily so --hf-mirror can set HF_ENDPOINT before huggingface_hub is loaded.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if is_local or (status is not None and not status.reachable):
        # Avoid a second long 443 timeout. For a remote model id this still
        # allows HuggingFace to resolve a complete pre-existing local cache.
        model_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True
    if device.startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    except Exception as exc:
        raise RuntimeError(
            _load_error_message(
                source=source, is_local=is_local, status=status, error=exc
            )
        ) from exc

    if device == "cpu":
        model = model.to("cpu")
    elif device == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer
