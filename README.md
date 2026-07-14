# Affective Substrate V2

Autonomous dynamical substrate for affective LLM modulation.

## Gate Results

| Gate | Status | Key Metric | Threshold |
|------|--------|------------|-----------|
| Gate A — Lyapunov chaos | **PASS** | λ_max = 0.01270 | λ > 0.01 |
| Gate A — boundedness | **PASS** | max \|x\| = 1.78028 | < 100 |
| Gate A — non-collapse | **PASS** | variance = 0.23003 | > 0.001 |
| Gate B — Input sensitivity | **PASS** | effect ratio = 505.3 | > 5× noise |
| Reservoir semantic probe | **PASS** | separation ratio = 2275.3 | > 2× |

Gate A used zero process noise, sane initial conditions near 0.5, and RK4. It
passed with the Hindmarsh–Rose fallback after the Mackey–Glass hybrid-kernel
sweep topped out at λ=0.008877, below the unchanged λ=0.01 threshold.

## Quick Start

```bash
cd /Users/alumin/Project/Affective_Substrate_V2
uv sync --group dev
uv run pytest
uv run python scripts/gateA_lyapunov.py
uv run python scripts/gateB_sensitivity.py
uv run python scripts/run_bare_substrate.py
```

## Phase 0 Exit

- `src/substrate/`: Mackey-Glass + hybrid kernel, Hindmarsh-Rose fallback, dual slow system, RK4 integrator.
- `src/memory/reservoir.py`: Echo-state network with `spectral_radius < 1`.
- `src/analysis/`: Lyapunov, sensitivity, reservoir probe, visualization.
- `scripts/gateA_lyapunov.py` and `scripts/gateB_sensitivity.py` both PASS.
- `artifacts/gate_results.json` records the gate metrics.
- `artifacts/kernel_sweep.json` records the Mackey–Glass fallback decision.

## Phase 1 — LLM Injection Pipeline

Phase 1 implements the LLM steering and injection machinery without requiring
the dynamical gates to be re-run. All new modules are independently tested and
written for the RTX 5090 Windows/CUDA 12.8 environment.

### New modules

- `src/llm/directions.py` — CAA extraction of affective steering directions
  from a HuggingFace model. Uses `model.model.layers[i]` forward hooks, no
  TransformerLens. Returns `V` of shape `[k, num_layers, d_model]`, optionally
  Gram–Schmidt orthogonalized per layer, and saves `artifacts/directions.pt` plus
  `artifacts/directions_meta.json`.
- `src/llm/hooks.py` — `InjectionHook` based on PyTorch `register_forward_hook`.
  Adds per-layer deltas to the residual stream, records `InjectionRecord` with
  raw/clipped norm, ratio to hidden, and clip flags.
- `src/llm/projection.py` — `compute_injection(state, r, V, P, config)` computes
  per-layer deltas:

  ```
  α_i = A_i · tanh(s_i · (x_i − x_baseline_i))
  β_i = B_i · tanh(s_i · (y_i − y_baseline_i))
  Δh_L = P·r + Σ_i α_i·V_i,L − Σ_i β_i·V_i,L
  ```

  Returns `dict[int, Tensor]` and an `InjectionRecord`. Clipping is applied via
  `config.cap` but the primary calibration knob is `A_i`.

### Scripts

- `scripts/extract_directions.py` — CLI that loads
  `Qwen/Qwen2.5-7B-Instruct`, extracts the four affective directions
  (`arousal`, `calm`, `positive`, `negative`), and saves artifacts.
  Supports `--device auto|cuda|cpu|mps`, `--dtype float16|bfloat16|float32`,
  and `--proxy http://127.0.0.1:7890` for Mac environments.
- `scripts/calibrate_injection.py` — CLI that loads saved directions and runs a
  fixed dummy prompt, sweeps `A_i` values, runs the model with injection
  enabled, and reports the `A_i` values that keep the raw 95th percentile norm
  below `cap`. Includes a `--device mock` mode for testing on machines without
  the 7B model.

### Tests

- `tests/test_projection.py` — dimension contracts, `InjectionRecord` contents,
  invalid-input validation, separate slow directions, and clipping behavior.
  No LLM weights are downloaded.

### Running on RTX 5090

The 7B model is **not** downloaded or run on the Mac. On the Windows 5090
workstation (CUDA 12.8, torch 2.11+cu128), install the project and run:

```bash
uv sync --group dev
uv run python scripts/extract_directions.py --device cuda --dtype bfloat16
uv run python scripts/calibrate_injection.py --device cuda --dtype bfloat16
```

If PyTorch wheels need to be installed manually for sm_120, see
`pyproject.toml` and install the CUDA 12.8 build before `uv sync`.

## Next Steps

Phase 2 will close the real feedback loop: add `src/feedback/sentiment.py`,
`src/llm/server.py`, and run Gate C with real LLM outputs.
