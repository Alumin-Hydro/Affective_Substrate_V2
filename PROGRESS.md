# Affective Substrate V2 — 开发追踪板

## 当前阶段

Phase 1 — LLM 注入管线（Mac mock/tiny 测试已通过，7B 留待 RTX 5090）

## 仓库

/Users/alumin/Project/Affective_Substrate_V2
SPEC: /Users/alumin/Project/Affective_Substrate_V2/SPEC.md

## Phase 0 状态（已完成）

- [x] 项目初始化（uv + git）
- [x] Codex 开发 Phase 0
- [x] Gate A PASS（Hindmarsh–Rose fallback）
- [x] Gate B PASS
- [x] git commit + 文件清单

## Phase 1 状态（已完成）

- [x] `src/llm/directions.py` — CAA 情感方向提取，支持正交化与元数据保存
- [x] `src/llm/hooks.py` — `InjectionHook`（仅 forward hooks，无 TransformerLens）
- [x] `src/llm/projection.py` — `compute_injection` + `InjectionRecord`
- [x] `scripts/extract_directions.py` — 针对 Qwen/Qwen2.5-7B-Instruct 的 CLI
- [x] `scripts/calibrate_injection.py` — A_i 标定 CLI，含 mock 模式
- [x] `tests/test_projection.py` — 维度契约与记录内容测试
- [x] 修复 mock 在 `torch.device("mock")` 处提前失败的问题
- [x] `InjectionHook` 支持无 decoder layers 的 mock/custom model，并提供幂等 `clear()` / `remove()`
- [x] 修复方向张量位置索引到真实注入层号的映射（如 0..3 → 15/18/21/24）
- [x] 提取/标定支持 `--local-model-dir`、`--hf-mirror`、`HF_ENDPOINT` 与网络诊断
- [x] 提取支持已完成产物复用，`--overwrite` 可强制重跑
- [x] `tests/test_calibrate_mock.py` — mock 主流程、tiny model hook、tuple 输出及断点复用
- [x] `tests/test_model_loading.py` — local/mirror/proxy、网络失败诊断与 fallback 文案
- [x] Mac `.venv`：`python -m pytest tests/ -q`（38 passed）
- [x] Mac `.venv`：`python scripts/calibrate_injection.py --device mock --max_new_tokens 4` 跑通
- [x] `pyproject.toml` — 添加 torch、transformers、accelerate
- [x] README.md / PROGRESS.md 更新
- [x] `.gitignore` — 排除 directions.pt 等生成物
- [ ] 在 RTX 5090 上运行 `extract_directions.py` 和 `calibrate_injection.py`（待后续）

## 说明

Phase 1 的离线 fallback 与 mock 标定已在 Mac 验证。7B 模型未在 Mac 下载或运行。
Windows RTX 5090（CUDA 12.8，torch 2.11+cu128）可通过 `--local-model-dir`
指向完整 Qwen snapshot；如镜像可用，也可使用 `--hf-mirror` 或 `HF_ENDPOINT`。
已有 `directions.pt` 与 `directions_meta.json` 时提取脚本会直接复用，不再加载模型。
