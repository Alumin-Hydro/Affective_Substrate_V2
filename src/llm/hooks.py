"""Residual-stream forward hooks for deterministic affective steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class InjectionRecord:
    """Per-layer diagnostics for one forward pass."""

    layer_norms_raw: dict[int, float] = field(default_factory=dict)
    layer_norms_clipped: dict[int, float] = field(default_factory=dict)
    ratio_to_hidden: dict[int, float] = field(default_factory=dict)
    clip_triggered: dict[int, bool] = field(default_factory=dict)

    def update(
        self,
        layer_idx: int,
        raw: float,
        clipped: float,
        ratio_to_hidden: float,
        clip_triggered: bool,
    ) -> None:
        self.layer_norms_raw[layer_idx] = raw
        self.layer_norms_clipped[layer_idx] = clipped
        self.ratio_to_hidden[layer_idx] = ratio_to_hidden
        self.clip_triggered[layer_idx] = clip_triggered


class InjectionHook:
    """Adds a delta vector to residual stream outputs at selected layers.

    The hook is registered with ``register_forward_hook`` on the model layers.
    It does not depend on TransformerLens or nnsight.
    """

    def __init__(self, model: nn.Module, inject_layers: list[int]) -> None:
        self.model = model
        self.inject_layers = sorted(inject_layers)
        self._deltas: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._last_record: InjectionRecord | None = None
        self._hook: Callable | None = None
        self.cap: float | None = None
        self._attach()

    def _attach(self) -> None:
        """Attach forward hooks to the configured layers."""

        layers = self._get_layers()
        if not layers:
            # Mock/custom models may intentionally expose no decoder-layer stack.
            # In that case the hook remains a safe no-op while retaining deltas.
            return
        for layer_idx in self.inject_layers:
            if layer_idx not in layers:
                raise ValueError(
                    f"layer {layer_idx} not available; model has {len(layers)} layers"
                )
            target = layers[layer_idx]
            handle = target.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _get_layers(self) -> dict[int, nn.Module]:
        """Discover the transformer decoder layers."""

        candidates = [
            getattr(getattr(self.model, "model", None), "layers", None),
            getattr(self.model, "layers", None),
            getattr(getattr(self.model, "transformer", None), "h", None),
            getattr(getattr(self.model, "gpt_neox", None), "layers", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                return {i: layer for i, layer in enumerate(candidate)}
        return {}

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            if layer_idx not in self._deltas:
                return output

            delta = self._deltas[layer_idx]
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                raise ValueError("hooked layer output does not contain a tensor")
            if hidden.dim() == 3:
                # [batch, seq, d_model] -> broadcast across seq
                delta_broadcast = delta.view(1, 1, -1).expand_as(hidden)
            elif hidden.dim() == 2:
                delta_broadcast = delta.view(1, -1).expand_as(hidden)
            else:
                raise ValueError(f"unexpected output shape: {hidden.shape}")

            modified = hidden + delta_broadcast.to(hidden.dtype).to(hidden.device)

            raw_norm = float(delta.norm(p=2).cpu())
            clipped_norm = raw_norm
            clip_triggered = False
            if hasattr(self, "cap") and self.cap is not None:
                if raw_norm > self.cap:
                    clipped_norm = float(self.cap)
                    clip_triggered = True

            hidden_norm = float(hidden.norm(p=2, dim=-1).mean().cpu())
            ratio = raw_norm / (hidden_norm + 1e-12)

            if self._last_record is None:
                self._last_record = InjectionRecord()
            self._last_record.update(
                layer_idx=layer_idx,
                raw=raw_norm,
                clipped=clipped_norm,
                ratio_to_hidden=ratio,
                clip_triggered=clip_triggered,
            )

            if isinstance(output, tuple):
                return (modified, *output[1:])
            if isinstance(output, list):
                return [modified, *output[1:]]
            return modified

        return hook

    def set_delta(self, layer_idx: int, delta: torch.Tensor) -> None:
        """Set the delta vector for ``layer_idx``.

        The delta is detached and moved to CPU; it will be cast to the layer
        output dtype/device during the forward pass.
        """

        if not isinstance(delta, torch.Tensor):
            raise TypeError("delta must be a torch.Tensor")
        if delta.ndim != 1:
            raise ValueError("delta must be a 1-D vector [d_model]")
        self._deltas[layer_idx] = delta.detach().cpu()

    def set_deltas(self, deltas: dict[int, torch.Tensor]) -> None:
        """Set deltas for multiple layers at once."""

        for layer_idx, delta in deltas.items():
            self.set_delta(layer_idx, delta)

    def clear(self) -> None:
        """Remove all pending deltas."""

        self._deltas.clear()
        self._last_record = None

    def remove(self) -> None:
        """Detach all forward hooks and clear pending state (idempotent)."""

        self._deltas.clear()
        self._last_record = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def last_record(self) -> InjectionRecord | None:
        return self._last_record

    @property
    def attached(self) -> bool:
        """Whether at least one real layer hook is currently registered."""

        return bool(self._handles)

    def __enter__(self) -> "InjectionHook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
