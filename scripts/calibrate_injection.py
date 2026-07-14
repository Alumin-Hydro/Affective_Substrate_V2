"""Calibrate A_i values so that raw 95th percentile injection norms stay below cap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.schema import InjectionConfig, LLMConfig, load_config
from src.llm.directions import load_directions
from src.llm.hooks import InjectionHook
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
    return parser.parse_args()


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


class MockLM:
    """Minimal mock LLM for calibration on machines without the real 7B model."""

    def __init__(self, d_model: int, num_layers: int, device: torch.device) -> None:
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device
        self.model = type(
            "MockModel",
            (),
            {
                "layers": [
                    type("MockLayer", (object,), {"forward": self._layer_forward})
                    for _ in range(num_layers)
                ]
            },
        )()

    def _layer_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

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


def load_model(llm_config: LLMConfig, device: str, dtype: torch.dtype, proxy: str | None):
    """Load a real or mock model depending on the requested device."""

    if proxy:
        import os

        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.setdefault(key, proxy)

    if device == "mock":
        raise ValueError("use build_mock_model instead of load_model for mock mode")

    device_arg = device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device_arg in ("cuda", "auto"):
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        llm_config.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_config.model_name, **model_kwargs
    )
    if device_arg == "cpu":
        model = model.to("cpu")
    elif device_arg == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tokenizer


def build_mock_model(llm_config: LLMConfig, device: torch.device) -> tuple:
    """Build a tokenizer-less mock model and a dummy tokenizer placeholder."""

    num_layers = max(llm_config.inject_layers) + 1
    d_model = getattr(llm_config, "d_model", 3584)
    model = MockLM(d_model=d_model, num_layers=num_layers, device=device)
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
    llm_config: LLMConfig,
    prompt: str,
    max_new_tokens: int,
) -> dict[int, list[float]]:
    """Run one prompt, injecting substrate-driven deltas at each token."""

    layer_norms: dict[int, list[float]] = {layer: [] for layer in llm_config.inject_layers}

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
        hook.set_deltas(deltas)
        _ = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )
        for layer in llm_config.inject_layers:
            layer_norms[layer].append(record.layer_norms_raw[layer])
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
    V = load_directions(args.directions)
    P = build_reservoir_projection(
        reservoir_dim=config.reservoir.dim, d_model=V.shape[2], seed=config.reservoir.seed
    )

    dtype = dtype_from_string(args.dtype)
    device_arg = args.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)

    if device_arg == "mock":
        model, tokenizer = build_mock_model(llm_config, device)
    else:
        model, tokenizer = load_model(llm_config, device_arg, dtype, args.proxy)

    hook = InjectionHook(model, inject_layers=llm_config.inject_layers)

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
            llm_config=llm_config,
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
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved calibration summary to {output_path}")

    hook.remove()


if __name__ == "__main__":
    main()
