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
source .venv/bin/activate
python -m pytest tests/ -q
python scripts/gateA_lyapunov.py
python scripts/gateB_sensitivity.py
python scripts/run_bare_substrate.py
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
  raw/clipped norm, ratio to hidden, and clip flags. Models without a recognized
  decoder-layer stack are supported as a safe no-op for mock/custom tests;
  `clear()` resets deltas and `remove()` detaches hooks.
- `src/llm/model_loading.py` — shared local-snapshot/mirror/proxy model loader.
  It probes the configured HuggingFace endpoint before remote loading and emits
  actionable offline fallback instructions if loading fails.
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
  `--local-model-dir`, `--hf-mirror`, and `--proxy`. When both output files
  already exist, it reuses them before loading a model; pass `--overwrite` to
  force extraction.
- `scripts/calibrate_injection.py` — CLI that loads saved directions and runs a
  fixed dummy prompt, sweeps `A_i` values, runs the model with injection
  enabled, and reports the `A_i` values that keep the raw 95th percentile norm
  below `cap`. It supports the same model-loading fallbacks. In `--device mock`
  mode, a missing directions artifact is replaced by deterministic synthetic
  directions so the complete calibration control flow can run offline.

### Tests

- `tests/test_projection.py` — dimension contracts, `InjectionRecord` contents,
  invalid-input validation, separate slow directions, and clipping behavior.
  No LLM weights are downloaded.
- `tests/test_calibrate_mock.py` — mock calibration CLI, no-layer mock hook,
  tiny PyTorch decoder hook, tuple outputs, cleanup, and extraction reuse.
- `tests/test_model_loading.py` — proxy/mirror environment setup, local snapshot
  validation, endpoint failure detection, and actionable error guidance.

### Running on macOS (offline/mock or small local model)

Use the repository virtual environment and CPU/float32 for Mac validation:

```bash
cd /Users/alumin/Project/Affective_Substrate_V2
source .venv/bin/activate
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890

python -m pytest tests/ -q
python scripts/calibrate_injection.py --device mock --dtype float32 --max_new_tokens 4
```

The mock command does not download weights. If `artifacts/directions.pt` is
absent, it reports that synthetic directions are being used and writes only
`artifacts/calibration_summary.json`.

The test suite also exercises a tiny in-memory PyTorch decoder. To smoke-test a
real small HuggingFace model already stored on disk, copy `config/default.yaml`
to a local config and set `llm.model_name` plus `llm.inject_layers` to layers
that exist in that model, then run both stages against the same snapshot:

```bash
python scripts/extract_directions.py \
  --config config/small-local.yaml \
  --local-model-dir /path/to/small-model-snapshot \
  --output_dir artifacts/small \
  --device cpu --dtype float32

python scripts/calibrate_injection.py \
  --config config/small-local.yaml \
  --local-model-dir /path/to/small-model-snapshot \
  --directions artifacts/small/directions.pt \
  --device cpu --dtype float32 --max_new_tokens 4
```

### Running on RTX 5090

The 7B model is **not** downloaded or run on the Mac. On the Windows 5090
workstation (CUDA 12.8, torch 2.11+cu128), use a complete local model snapshot
when direct HuggingFace 443 access is unavailable:

```powershell
.venv\Scripts\python.exe scripts\extract_directions.py `
  --local-model-dir D:\models\Qwen2.5-7B-Instruct `
  --device cuda --dtype bfloat16

.venv\Scripts\python.exe scripts\calibrate_injection.py `
  --local-model-dir D:\models\Qwen2.5-7B-Instruct `
  --device cuda --dtype bfloat16
```

`--local-model-dir` must point to the model/snapshot directory containing
`config.json`, tokenizer files, and all weight shards—not merely the parent HF
cache directory. Extraction skips model loading when both `directions.pt` and
`directions_meta.json` already exist. Use `--overwrite` only when a fresh
extraction is intended.

If a mirror is reachable, either form below is supported:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
.venv\Scripts\python.exe scripts\extract_directions.py --device cuda --dtype bfloat16

# Equivalent one-command form; an explicit URL may follow --hf-mirror.
.venv\Scripts\python.exe scripts\extract_directions.py --hf-mirror --device cuda --dtype bfloat16
```

Remote loading performs a short endpoint check. A failed load reports whether
the endpoint was reachable and suggests `--local-model-dir`, `--hf-mirror` /
`HF_ENDPOINT`, and `HTTPS_PROXY` / `--proxy` as concrete recovery options.

If PyTorch wheels need to be installed manually for sm_120, see
`pyproject.toml` and install the CUDA 12.8 build before `uv sync`.

## Next Steps

Phase 2 will close the real feedback loop: add `src/feedback/sentiment.py`,
`src/llm/server.py`, and run Gate C with real LLM outputs.
