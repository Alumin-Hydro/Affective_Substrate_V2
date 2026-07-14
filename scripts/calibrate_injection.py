"""Calibrate A_i values so that raw 95th percentile injection norms stay below cap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.schema import InjectionConfig, LLMConfig, load_config
from src.llm.directions import load_directions
from src.llm.hooks import InjectionHook
from src.llm.model_loading import DEFAULT_HF_MIRROR, load_causal_lm
from src.llm.projection import build_reservoir_projection, compute_injection
from src.memory.reservoir import Reservoir
from src.substrate import create_core


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

DEFAULT_PROMPT = "The user just said: hello. Respond briefly."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate A_i gains for affective injection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="artifacts/directions.pt",
        help="Path to extracted directions.pt file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Dummy prompt used during calibration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu', 'mps', or 'mock'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--A_sweep",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated A_i values to sweep (applied to all dims)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Number of tokens to generate for norm measurement",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional HTTP/HTTPS proxy URL for downloads",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default=None,
        help="Complete local HuggingFace model/snapshot directory (no network needed)",
    )
    parser.add_argument(
        "--hf-mirror",
        nargs="?",
        const=DEFAULT_HF_MIRROR,
        default=None,
        metavar="URL",
        help=f"Use an HF mirror; with no URL defaults to {DEFAULT_HF_MIRROR}",
    )
    return parser.parse_args()


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


class MockLM:
    """Minimal mock LLM for calibration on machines without the real 7B model."""

    def __init__(self, d_model: int, device: torch.device) -> None:
        self.d_model = d_model
        self.device = device

    def __call__(self, **kwargs):
        batch_size = kwargs.get("input_ids", torch.zeros(1, 1)).shape[0]
        seq_len = kwargs.get("input_ids", torch.zeros(1, 1)).shape[1]
        hidden = torch.randn(
            batch_size, seq_len, self.d_model, device=self.device, dtype=torch.float32
        )
        return type("MockOutput", (), {"logits": hidden, "hidden_states": None})()

    def generate(self, input_ids, **kwargs):
        max_new = kwargs.get("max_new_tokens", 1)
        return torch.cat(
            [input_ids, torch.zeros(input_ids.shape[0], max_new, dtype=torch.long, device=self.device)],
            dim=1,
        )

    def parameters(self):
        yield torch.zeros(1, device=self.device, requires_grad=True)


def load_model(
    llm_config: LLMConfig,
    device: str,
    dtype: torch.dtype,
    proxy: str | None = None,
    local_model_dir: str | None = None,
    hf_mirror: str | None = None,
):
    """Load a real or mock model depending on the requested device."""

    if device == "mock":
        raise ValueError("use build_mock_model instead of load_model for mock mode")

    return load_causal_lm(
        model_name=llm_config.model_name,
        device=device,
        dtype=dtype,
        local_model_dir=local_model_dir,
        hf_mirror=hf_mirror,
        proxy=proxy,
    )


def build_mock_model(device: torch.device, d_model: int) -> tuple:
    """Build a tokenizer-less mock model and a dummy tokenizer placeholder."""

    model = MockLM(d_model=d_model, device=device)
    tokenizer = type(
        "MockTokenizer",
        (object,),
        {
            "encode": lambda prompt, **kw: [0, 1, 2],
            "__call__": lambda self, text, **kw: {
                "input_ids": torch.tensor([[0, 1, 2]], device=device)
            },
            "pad_token_id": 0,
        },
    )()
    return model, tokenizer


def build_mock_directions(k: int, num_layers: int, d_model: int = 64) -> np.ndarray:
    """Create deterministic normalized directions when no artifact exists in mock mode."""

    rng = np.random.default_rng(0)
    directions = rng.normal(size=(k, num_layers, d_model))
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return (directions / np.maximum(norms, 1e-12)).astype(np.float64)


def _tokens_from_model(
    model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int
) -> list[int]:
    """Generate a few tokens with deterministic decoding."""

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
    )
    return generated[0].tolist()


def _run_calibration_trial(
    model,
    tokenizer,
    hook: InjectionHook,
    substrate_core,
    reservoir,
    V: np.ndarray,
    P: np.ndarray,
    injection_config: InjectionConfig,
    inject_layers: list[int],
    prompt: str,
    max_new_tokens: int,
) -> dict[int, list[float]]:
    """Run one prompt, injecting substrate-driven deltas at each token."""

    layer_norms: dict[int, list[float]] = {layer: [] for layer in inject_layers}

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    for _ in range(max_new_tokens):
        substrate_core.step(dt=0.1)
        reservoir.step(dt=0.1, x=substrate_core.state.x, u=None)
        deltas, record = compute_injection(
            state=substrate_core.state,
            r=reservoir.state,
            V=V,
            P=P,
            config=injection_config,
        )
        if len(deltas) != len(inject_layers):
            raise ValueError(
                "direction layer count does not match configured injection layer count"
            )
        hook.set_deltas(
            {
                model_layer: deltas[position]
                for position, model_layer in enumerate(inject_layers)
            }
        )
        _ = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )
        for position, layer in enumerate(inject_layers):
            layer_norms[layer].append(record.layer_norms_raw[position])
        input_ids = torch.tensor(
            [_tokens_from_model(model, tokenizer, input_ids, 1)], device=input_ids.device
        )

    hook.clear()
    return layer_norms


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    llm_config = config.llm
    injection_config = llm_config.injection
    inject_layers = sorted(llm_config.inject_layers)

    mock_mode = args.device == "mock"
    device_arg = "cpu" if mock_mode else args.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)

    directions_path = Path(args.directions)
    if directions_path.exists():
        V = load_directions(directions_path)
    elif mock_mode:
        V = build_mock_directions(
            k=config.substrate.k, num_layers=len(inject_layers)
        )
        print(
            f"Directions file {directions_path} was not found; using deterministic "
            f"mock directions with shape {V.shape}."
        )
    else:
        raise FileNotFoundError(
            f"directions file not found: {directions_path}. Run extract_directions.py "
            "first or pass --directions PATH."
        )
    if V.shape[0] != config.substrate.k:
        raise ValueError(
            f"directions contain k={V.shape[0]}, expected {config.substrate.k}"
        )
    if V.shape[1] != len(inject_layers):
        raise ValueError(
            f"directions contain {V.shape[1]} layers, expected "
            f"{len(inject_layers)}"
        )

    substrate_core = create_core(config.substrate)
    reservoir = Reservoir(
        dim=config.reservoir.dim,
        spectral_radius=config.reservoir.spectral_radius,
        input_scaling=config.reservoir.input_scaling,
        leak_rate=config.reservoir.leak_rate,
        tau_r=config.reservoir.tau_r,
        x_dim=config.reservoir.x_dim,
        input_dim=config.reservoir.input_dim,
        density=config.reservoir.density,
        seed=config.reservoir.seed,
    )
    P = build_reservoir_projection(
        reservoir_dim=config.reservoir.dim, d_model=V.shape[2], seed=config.reservoir.seed
    )

    dtype = dtype_from_string(args.dtype)
    if mock_mode:
        model, tokenizer = build_mock_model(device, d_model=V.shape[2])
    else:
        model, tokenizer = load_model(
            llm_config,
            device_arg,
            dtype,
            proxy=args.proxy,
            local_model_dir=args.local_model_dir,
            hf_mirror=args.hf_mirror,
        )

    hook = InjectionHook(model, inject_layers=inject_layers)

    A_values = [float(x.strip()) for x in args.A_sweep.split(",") if x.strip()]
    cap = injection_config.cap

    summary = []
    for A in A_values:
        trial_config = InjectionConfig(
            A=[A] * config.substrate.k,
            B=injection_config.B,
            s=injection_config.s,
            x_baseline=injection_config.x_baseline,
            y_baseline=injection_config.y_baseline,
            cap=cap,
            slow_use_separate_directions=injection_config.slow_use_separate_directions,
        )

        substrate_core.reset()
        reservoir.reset()
        layer_norms = _run_calibration_trial(
            model=model,
            tokenizer=tokenizer,
            hook=hook,
            substrate_core=substrate_core,
            reservoir=reservoir,
            V=V,
            P=P,
            injection_config=trial_config,
            inject_layers=inject_layers,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )

        per_layer_95 = {
            layer: float(np.percentile(norms, 95))
            for layer, norms in layer_norms.items()
        }
        max_95 = max(per_layer_95.values()) if per_layer_95 else 0.0
        under_cap = max_95 < cap
        summary.append({
            "A": A,
            "per_layer_95th": per_layer_95,
            "max_95th": max_95,
            "under_cap": under_cap,
        })
        print(f"A={A:6.3f}: max 95th raw norm = {max_95:6.3f}, cap={cap:6.3f}, {'PASS' if under_cap else 'FAIL'}")

    print("\nRecommended A values (raw 95th < cap):")
    for entry in summary:
        if entry["under_cap"]:
            print(f"  A = {entry['A']}")

    output_path = Path(args.directions).parent / "calibration_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved calibration summary to {output_path}")

    hook.remove()


if __name__ == "__main__":
    main()
