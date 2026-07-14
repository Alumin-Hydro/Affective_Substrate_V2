# Affective Substrate V2

Autonomous dynamical substrate for affective LLM modulation. Phase 0 implements the substrate core, validation gates, and reservoir — no LLM hook yet.

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

## Next Steps

Phase 1 code is intentionally not implemented in this Phase 0 commit.
