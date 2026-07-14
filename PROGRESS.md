# Affective Substrate V2 — 开发追踪板

## 当前阶段

Phase 1 — LLM 注入管线（已提交，未在 Mac 运行 7B 模型）

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
- [x] `pyproject.toml` — 添加 torch、transformers、accelerate
- [x] README.md / PROGRESS.md 更新
- [x] `.gitignore` — 排除 directions.pt 等生成物
- [ ] 在 RTX 5090 上运行 `extract_directions.py` 和 `calibrate_injection.py`（待后续）

## 说明

Phase 1 代码已按 SPEC 5.3 与 7 实现，但 7B 模型未在 Mac 本地下载或运行。
实际提取与标定需在 Windows RTX 5090 工作站（CUDA 12.8，torch 2.11+cu128）上执行。
