You are implementing Affective Substrate V2 from the SPEC at /Users/alumin/Project/Affective_Substrate_V2/SPEC.md. Read it carefully first.

Repository root: /Users/alumin/Project/Affective_Substrate_V2
It is already initialized with `uv` for Python 3.11 and `git`.

Your task: implement Phase 0 ONLY. Do NOT write any llm/ or feedback/ code yet. Phase 0 gates must PASS before anything else.

Phase 0 deliverables:
1. Implement `src/substrate/base.py`, `src/substrate/state.py`, `src/substrate/kernels.py`, `src/substrate/mackey_glass.py`, `src/substrate/dual_system.py`, `src/substrate/hindmarsh_rose.py` (fallback).
2. Implement `src/memory/reservoir.py`.
3. Implement `src/analysis/lyapunov.py`, `src/analysis/sensitivity.py`, `src/analysis/reservoir_probe.py`, `src/analysis/viz.py`.
4. Implement `scripts/gateA_lyapunov.py`, `scripts/gateB_sensitivity.py`, `scripts/sweep_kernel.py`, `scripts/run_bare_substrate.py`.
5. Implement `tests/test_kernels.py`, `tests/test_substrate.py`, `tests/test_reservoir.py`, `tests/test_config.py`.
6. Implement `config/schema.py` (Pydantic v2), `config/default.yaml`, `config/phase0.yaml`.
7. Implement `pyproject.toml` with all Phase 0 deps (numpy, scipy, matplotlib, pyyaml, pydantic>=2).
8. Run `scripts/gateA_lyapunov.py` and `scripts/gateB_sensitivity.py`. If Gate A FAILS, run `scripts/sweep_kernel.py` over `w` and `alpha`, and if still failing switch `core` to `hindmarsh_rose` in `config/phase0.yaml` and rerun. Do not proceed until both gates PASS.
9. Write results to `artifacts/gate_results.json`.
10. Update `README.md` with a gate results table and quick start.

Strict rules from SPEC:
- Sane initial conditions (~0.5), no hot IC.
- Gate A: noise_sigma must be 0. lambda_min = 0.01, blowup_threshold = 100, collapse_threshold = 0.001.
- Gate B: dist(emotional, neutral) > 5 * dist(noise_baseline).
- Each module must be independently testable by tests/.
- Do not adjust gate thresholds to pass; fix the system instead.
- Use RK4 integrator by default.
- Use Python 3.11+, uv, no global pollution.
- Keep all files under `/Users/alumin/Project/Affective_Substrate_V2`.
- When running git/curl/uv invocations that need to reach github.com, set `HTTPS_PROXY=http://127.0.0.1:7890` and `HTTP_PROXY=http://127.0.0.1:7890` in the environment.

After you finish, run the full test suite with `uv run pytest` and report the gate results, test pass/fail counts, and a list of all created/modified files. Then commit to git with a clear message.

Begin by reading SPEC.md, then execute Phase 0 in order.
