"""Offline CAA extraction of affective steering directions from Qwen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.schema import load_config
from src.llm.directions import DIMS, extract_directions, save_directions
from src.llm.model_loading import DEFAULT_HF_MIRROR, load_causal_lm


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CAA steering directions from a HuggingFace model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory where directions.pt and directions_meta.json will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load the model on: 'auto', 'cuda', 'cpu', or 'mps'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for extraction",
    )
    parser.add_argument(
        "--orthogonalize",
        type=int,
        default=1,
        help="1 to orthogonalize directions per layer, 0 otherwise",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional HTTP/HTTPS proxy URL for downloads (e.g. http://127.0.0.1:7890)",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run extraction even when both output artifacts already exist",
    )
    return parser.parse_args()


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions_path = output_dir / "directions.pt"
    metadata_path = output_dir / "directions_meta.json"
    if directions_path.exists() and metadata_path.exists() and not args.overwrite:
        print(
            f"Reusing existing extraction artifacts: {directions_path} and "
            f"{metadata_path}. Pass --overwrite to extract again."
        )
        return

    config = load_config(args.config)
    llm_config = config.llm

    dtype = dtype_from_string(args.dtype)
    device_arg = args.device
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    source = args.local_model_dir or llm_config.model_name
    print(f"Loading model {source} on {device_arg} with dtype {dtype} ...")
    model, tokenizer = load_causal_lm(
        model_name=llm_config.model_name,
        device=device_arg,
        dtype=dtype,
        local_model_dir=args.local_model_dir,
        hf_mirror=args.hf_mirror,
        proxy=args.proxy,
    )
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    layers = sorted(llm_config.inject_layers)
    print(f"Extracting directions for layers {layers}")
    V = extract_directions(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        orthogonalize=bool(args.orthogonalize),
        device=device,
    )
    print(f"Extracted directions with shape {V.shape}")

    save_directions(
        output_dir=output_dir,
        directions=V,
        layers=layers,
        dim_names=DIMS,
        orthogonalize=bool(args.orthogonalize),
    )
    print(f"Saved directions to {output_dir / 'directions.pt'}")
    print(f"Saved metadata to {output_dir / 'directions_meta.json'}")


if __name__ == "__main__":
    main()
