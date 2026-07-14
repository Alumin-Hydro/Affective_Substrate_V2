# Affective Substrate V2 — 开发追踪板

## 当前阶段
Phase 0 — Substrate 验证（无 LLM）

## 目标
实现 Mackey-Glass + 混合延迟核 + 快慢双系统 + reservoir，并通过：
- Gate A: Lyapunov λ_max > 0.01（自主混沌，无噪声）
- Gate B: 输入敏感性（强情感 vs 中性轨迹差异 > 5× 噪声基线）

## 仓库
/Users/alumin/Project/Affective_Substrate_V2
SPEC: /Users/alumin/Project/Affective_Substrate_V2/SPEC.md

## 状态
- [x] 项目初始化（uv + git）
- [x] Codex 开发 Phase 0
- [x] Gate A PASS（Hindmarsh–Rose fallback）
- [x] Gate B PASS
- [ ] git commit + 文件清单
