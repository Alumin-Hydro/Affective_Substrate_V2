"""Affective steering-direction extraction using CAA on a HuggingFace model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


DIMS = ["arousal", "calm", "positive", "negative"]
DEFAULT_CONTRAST_PAIRS = [
    # (positive_prompt, negative_prompt) for each affective dimension
    ([
        "I feel wired, energetic, and ready to act.",
        "Adrenaline is surging through me; I am highly alert.",
    ], [
        "I feel sluggish, sleepy, and completely unenergetic.",
        "My body is relaxed to the point of drowsiness.",
    ]),
    ([
        "I am calm, peaceful, and completely at ease.",
        "My breathing is slow and steady; nothing can disturb me.",
    ], [
        "I am tense, jittery, and on edge about everything.",
        "My mind is racing with worry and restlessness.",
    ]),
    ([
        "I feel happy, optimistic, and full of joy.",
        "Today is wonderful and everything seems bright.",
    ], [
        "I feel sad, hopeless, and defeated.",
        "Nothing good ever happens and the future looks bleak.",
    ]),
    ([
        "I feel angry, frustrated, and filled with negativity.",
        "Everything annoys me and I want to push back.",
    ], [
        "I feel content, forgiving, and free of resentment.",
        "I accept what happened and hold no grudge.",
    ]),
]


def _validate_contrast_pairs(
    contrast_pairs: list[tuple[list[str], list[str]]],
) -> None:
    if len(contrast_pairs) != 4:
        raise ValueError("contrast_pairs must contain 4 (positive, negative) groups")
    for dim_idx, (pos, neg) in enumerate(contrast_pairs):
        if not pos or not neg:
            raise ValueError(f"dimension {dim_idx} has empty positive or negative list")


def get_default_contrast_pairs() -> list[tuple[list[str], list[str]]]:
    """Return the default CAA contrast pairs for [arousal, calm, positive, negative]."""

    return [tuple(pair) for pair in DEFAULT_CONTRAST_PAIRS]


def _hidden_states_for_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int] | None,
    device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    """Compute per-layer last-token hidden states for a list of prompts."""

    layers = list(range(len(model.model.layers))) if layers is None else layers
    layer_to_states: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []
    hooks = {}

    def make_hook(layer_index: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_to_states[layer_index].append(hidden[:, -1, :].detach().cpu())
            if isinstance(output, tuple):
                return output
            return output

        return hook

    for layer_index in layers:
        target = model.model.layers[layer_index]
        hooks[layer_index] = target.register_forward_hook(make_hook(layer_index))
        handles.append(hooks[layer_index])

    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return layer_to_states


def _mean_direction(
    pos_states: list[torch.Tensor], neg_states: list[torch.Tensor]
) -> torch.Tensor:
    """Average positive-minus-negative difference and normalize."""

    pos = torch.stack(pos_states).mean(dim=0)
    neg = torch.stack(neg_states).mean(dim=0)
    direction = pos - neg
    norm = direction.norm(p=2)
    if norm < 1e-12:
        return direction
    return direction / norm


def orthogonalize_directions(
    directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Gram--Schmidt orthogonalize [k, d_model] directions."""

    directions = np.asarray(directions, dtype=np.float64)
    out = np.zeros_like(directions)
    for i in range(directions.shape[0]):
        v = directions[i].copy()
        for j in range(i):
            proj = np.dot(v, out[j]) * out[j]
            v = v - proj
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
        out[i] = v
    return out


def extract_directions(
    model: torch.nn.Module,
    tokenizer: Any,
    contrast_pairs: list[tuple[list[str], list[str]]] | None = None,
    layers: list[int] | None = None,
    orthogonalize: bool = True,
    device: torch.device | None = None,
) -> NDArray[np.float64]:
    """Extract CAA steering directions from a HuggingFace model.

    Returns
    -------
    directions
        Array of shape ``[k, num_layers, d_model]`` where k is the number of
        affective dimensions (4 by default).
    """

    if contrast_pairs is None:
        contrast_pairs = get_default_contrast_pairs()
    _validate_contrast_pairs(contrast_pairs)

    if device is None:
        device = next(model.parameters()).device
    if layers is None:
        layers = list(range(len(model.model.layers)))

    directions_per_layer: dict[int, torch.Tensor] = {}
    for dim_idx, (pos_prompts, neg_prompts) in enumerate(contrast_pairs):
        pos_states = _hidden_states_for_prompts(
            model, tokenizer, pos_prompts, layers, device
        )
        neg_states = _hidden_states_for_prompts(
            model, tokenizer, neg_prompts, layers, device
        )
        for layer in layers:
            direction = _mean_direction(pos_states[layer], neg_states[layer])
            if layer not in directions_per_layer:
                directions_per_layer[layer] = []
            directions_per_layer[layer].append(direction)

    k = len(contrast_pairs)
    d_model = directions_per_layer[layers[0]][0].shape[0]
    V = np.zeros((k, len(layers), d_model), dtype=np.float64)
    for layer_idx, layer in enumerate(layers):
        for dim_idx in range(k):
            V[dim_idx, layer_idx, :] = directions_per_layer[layer][dim_idx].numpy()

    if orthogonalize:
        for layer_idx in range(V.shape[1]):
            V[:, layer_idx, :] = orthogonalize_directions(V[:, layer_idx, :])

    return V


def save_directions(
    output_dir: str | Path,
    directions: NDArray[np.float64],
    layers: list[int],
    dim_names: list[str] | None = None,
    orthogonalize: bool = True,
) -> None:
    """Save direction tensor and JSON metadata to ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dim_names = dim_names or DIMS
    if len(dim_names) != directions.shape[0]:
        raise ValueError("dim_names length must match number of directions")

    torch.save(torch.from_numpy(directions), output_dir / "directions.pt")

    V = directions
    cos_sim = np.zeros((V.shape[0], V.shape[0]))
    for i in range(V.shape[0]):
        for j in range(V.shape[0]):
            vi = V[i].reshape(-1)
            vj = V[j].reshape(-1)
            cos_sim[i, j] = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12))

    metadata = {
        "shape": list(V.shape),
        "layers": layers,
        "dim_names": dim_names,
        "orthogonalize": orthogonalize,
        "cosine_similarity": cos_sim.tolist(),
    }
    with open(output_dir / "directions_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_directions(path: str | Path) -> NDArray[np.float64]:
    """Load directions from a ``directions.pt`` file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"directions file not found: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor.numpy().astype(np.float64)


def load_directions_meta(path: str | Path) -> dict[str, Any]:
    """Load metadata from a ``directions_meta.json`` file."""

    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
