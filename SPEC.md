# 情感基底层系统 v2 — 研发规格说明书 (SPEC)

> **文档性质**：面向 coding agent（Codex）的实现规格。人类可读、agent 可执行。
> **实现语言**：Python 3.11+，PyTorch。所有标识符、文件名、config key 用英文；解释性说明用中文。
> **给 Codex 的总原则**：严格按 Phase 顺序实现。**Phase 0 的三道 gate 未全部通过，禁止进入 Phase 1。** 每个模块必须能被对应的 `tests/` 独立测试。不要为了"跑通"而放宽 gate 阈值——阈值就是本项目的科学价值所在。

---

## 0. 这一版要修正什么（v1 尸检报告）

v1 已经能跑，架构完整，但缺了让系统对自己负责的实验。v2 的全部设计约束都来自 v1 的三个致命教训：

| # | v1 的问题 | 证据 | v2 的对冲 |
|---|---|---|---|
| **T1** | **幂律核把混沌杀死了**。substrate 收敛到不动点，"动态感"全来自外部扰动而非系统自身。 | Delay Kernel 对比图：幂律/指数核都收敛到 1.0，只有 fixed-τ 保持振荡。 | 混合核（Dirac+幂律）；核作为可替换模块；**Gate A 强制验证 Lyapunov λ>0**。 |
| **T2** | **注入范数三组数据互相矛盾**，且疑似长期顶到 cap=10。方向信息被 clip 抹掉，注入退化成开关式 on/off。 | 全景图 bar 恒在 ~9.5；监控图均值 0.53；报告文字均值 5.31。 | 注入强度作为**受控实验变量**；raw norm（clip 前）per-layer 直方图从第一天记录；主调节旋钮是 `A_i` 而非 cap。 |
| **T3** | **"情感自然衰减"可能是 illusion**。起始 IC=3.55 远高于工作点 ~1.0，轨迹很可能只是过热 IC 回弹，看不到对话内容的真实影响。 | 状态从 [3.55,...] 单调衰减到 [1.03,...]，任何阻尼系统从 hot IC 都会这样。 | IC 强制在 sane 范围（~0.5）；**Gate B 验证输入敏感性**；**Gate C 验证反馈因果性**。 |

**一句话总结 v2 的哲学**：v1 证明了"能搭出骨架"，v2 要证明"骨架真的活着、真的在呼吸、真的对内容敏感"。

---

## 1. 不可动摇的原则

1. **substrate 独立于 LLM 存在**。它是一个自主动力系统。LLM 只是通过 activation injection *读取* 它的状态。断开 LLM，substrate 仍在运动。
2. **记忆携带在动力状态里，不塞进 context**。这是与 RAG 的本质区别。记忆通过塑造 `x, y, r` 的状态间接影响输出，模型永远看不到符号化的记忆条目。
3. **可证伪优先于好看**。任何"看起来像情感"的现象，必须能通过消融/对照实验证明它不是 artifact（IC 回弹、噪声、外部扰动）。
4. **注入强度是变量不是常数**。每一步的 raw 注入范数必须可观测、可复现、可扫描。
5. **Sane initial conditions**。所有状态初始化在自然工作点附近（~0.5），禁止 hot IC。

---

## 2. 核心设计决策（含理由）

### D1 · 动力核：Mackey-Glass + 混合延迟核，核可替换

**决策**：主实现用 Mackey-Glass，延迟项用混合核：

```
K_delayed_i(t) = w · x_i(t − τ_i) + (1 − w) · ∫₀^∞ K_pl(s) · x_i(t − s) ds
```

- `w=1` → 退化为纯 MG（混沌）；`w=0` → 纯幂律（v1 的不动点）。**中间必存在混沌窗口**，`w` 是关键扫描参数。
- Dirac 分量（`w` 部分）负责维持振荡基础；幂律分量负责长程记忆（重尾）。

**理由**：MG 语义最贴"激素/生理控制"（1977 年就是为白细胞调节推的），保留隐喻。混合核直接对冲 T1。

**关键对冲**：`src/substrate/base.py` 定义抽象接口 `DynamicalCore`，Mackey-Glass 和 Hindmarsh-Rose 都实现它。**如果 Gate A 用 MG+混合核在任何 `w` 下都拿不到 λ>0，切换到 Hindmarsh-Rose（天然混沌+bursting，快慢结构方程自带）——一行 config 切换，不改上层。** 这是本项目最大不确定性的保险。

### D2 · 快慢双系统 + 自适应阻尼

保留 v1 唯一验证成功的机制（图2 的 γ_eff 跟随 y(t) 吻合得很漂亮）：

```
dy_i/dt = ε_i · (|x_i| − y_i)        # 慢系统 = |x| 的低通滤波，ε_i ≪ 1/τ_i
γ_i(y_i) = γ_0 + κ · y_i            # 阻尼随慢系统自适应
```

y 增大 → 阻尼增大 → x 自动镇静。这是"副交感"。

### D3 · 注入定标作为一等公民

不再有"事后监控"。`src/llm/projection.py` 每次 `compute_injection()` 必须返回一个 `InjectionRecord`（见 5.3），包含 per-layer 的 raw norm（clip 前）、clip 后 norm、`‖Δh‖/‖h‖` 比值。主调节旋钮是每维的 `A_i`，标定目标：**raw norm 直方图的 95 分位 < cap**，让 clip 几乎不触发。

### D4 · 反馈用分类器不用词典

v1 的中文情感词典对 LLM 输出（比喻、反讽、长句）太粗糙。v2 用小型 sentiment/VAD 分类器（见 5.4），词典仅作 fallback。

### D5 · 分层记忆但延后验证

Reservoir（L1）在 Phase 0 就要，但必须验证它编码的是语义而非噪声（Gate B 的副产物 + 专项测试）。Hopfield（L2）和 slow weights（L3）推到 Phase 3，用 z-score 触发替代 v1 的固定阈值 0.3。

---

## 3. 系统架构总览

```
                          ┌─────────────────────────────────────┐
        Phase 3 (later)   │  L3 slow weights  W_r, V (Hebbian)   │  小时-天 / 性格
                          ├─────────────────────────────────────┤
        Phase 3 (later)   │  L2 Hopfield episodic  M={(r,x,y)_k} │  情节 / z-score 触发
                          ├─────────────────────────────────────┤
        Phase 0           │  L1 reservoir  r(t)  (echo state)    │  秒-分钟 / 工作记忆
                          ├─────────────────────────────────────┤
        Phase 0 ← 核心    │  dual system  x(t) fast / y(t) slow  │  交感 / 副交感
                          │  Mackey-Glass + hybrid delay kernel  │
                          └──────────────┬──────────────────────┘
                                         │ state → projection → Δh
        Phase 1                          ↓
                          ┌─────────────────────────────────────┐
                          │  LLM (Qwen)  residual stream inject  │  皮层
                          │  hooks @ layers L, deterministic     │
                          └──────────────┬──────────────────────┘
                                         │ output text
        Phase 2                          ↓
                          ┌─────────────────────────────────────┐
                          │  sentiment/VAD classifier → feedback │  闭环
                          └──────────────┬──────────────────────┘
                                         └──→ external_input → substrate
```

数据流（一个 turn）：`user input → LLM(+当前 substrate 注入) → output → VAD → external_input → substrate.step() ×N → 新状态 → 下一 turn 的注入`。

---

## 4. 文件树

```
affective-substrate/
├── SPEC.md                      # 本文档
├── README.md                    # 快速上手 + gate 结果表
├── pyproject.toml               # 依赖锁版本（见 §9）
├── config/
│   ├── default.yaml             # 全量默认参数
│   ├── phase0.yaml              # 混沌调参专用（无 LLM）
│   └── schema.py                # pydantic config 模型 + 校验
├── src/
│   ├── substrate/
│   │   ├── base.py              # DynamicalCore 抽象接口
│   │   ├── mackey_glass.py      # 主实现
│   │   ├── hindmarsh_rose.py    # 保险实现（Gate A 兜底）
│   │   ├── kernels.py           # dirac / powerlaw / hybrid 延迟核
│   │   ├── dual_system.py       # 快慢耦合 + 自适应阻尼 + 耦合矩阵 W
│   │   └── state.py             # SubstrateState dataclass
│   ├── memory/
│   │   ├── reservoir.py         # echo state network (L1)
│   │   ├── hopfield.py          # modern Hopfield episodic (L2, Phase 3)
│   │   └── slow_weights.py      # Hebbian slow plasticity (L3, Phase 3)
│   ├── llm/
│   │   ├── directions.py        # CAA 情感方向提取
│   │   ├── hooks.py             # residual stream forward hooks
│   │   ├── projection.py        # state → steering vector + InjectionRecord
│   │   └── server.py            # HTTP API (Phase 2)
│   ├── feedback/
│   │   ├── sentiment.py         # VAD 分类器 (+ 词典 fallback)
│   │   └── loop.py              # 闭环控制器
│   └── analysis/
│       ├── lyapunov.py          # Gate A：最大 Lyapunov 指数
│       ├── sensitivity.py       # Gate B：输入敏感性
│       ├── ablation.py          # Gate C：反馈因果性（harness 早建）
│       ├── reservoir_probe.py   # reservoir 语义编码验证
│       └── viz.py               # 全部可视化
├── scripts/
│   ├── gateA_lyapunov.py        # 跑 Gate A，输出 PASS/FAIL + λ
│   ├── gateB_sensitivity.py     # 跑 Gate B
│   ├── gateC_causality.py       # 跑 Gate C（Phase 2）
│   ├── sweep_kernel.py          # 扫 w / alpha，找混沌窗口
│   ├── extract_directions.py    # 离线提取情感方向 V
│   ├── calibrate_injection.py   # 标定 A_i，输出 raw norm 直方图
│   ├── run_bare_substrate.py    # 无 LLM 裸跑 + 可视化
│   └── run_full_system.py       # 完整系统 + 交互
├── tests/
│   ├── test_kernels.py
│   ├── test_substrate.py        # 含数值稳定性、能量有界性
│   ├── test_reservoir.py        # echo state property (spectral radius < 1)
│   ├── test_projection.py       # 维度契约 + norm 记录
│   └── test_config.py
└── artifacts/                   # gate 结果、图、标定数据（git-ignore 大文件）
    ├── gate_results.json
    └── figures/
```

---

## 5. 模块契约

> 每个契约给出：职责、关键接口签名、输入输出、以及必须满足的不变量（invariant）。

### 5.1 substrate

**`state.py`**
```python
@dataclass
class SubstrateState:
    x: np.ndarray          # shape [k], 快系统（情感维度：arousal, calm, positive, negative）
    y: np.ndarray          # shape [k], 慢系统（基线/激素稳态）
    gamma_eff: np.ndarray  # shape [k], 当前有效阻尼（诊断用）
    t: int                 # 当前步数
    history: np.ndarray    # shape [buffer_len, k], 延迟核所需的历史缓冲
```

**`base.py` — `DynamicalCore` 抽象接口**（所有核必须实现）
```python
class DynamicalCore(ABC):
    @abstractmethod
    def step(self, dt: float, external_input: np.ndarray | None = None) -> None: ...
    #   推进一步。external_input shape [k] 或 None（自主运行）。
    #   不变量：调用后 self.state.t += 1；history 缓冲正确滚动。

    @abstractmethod
    def reset(self, ic: np.ndarray | None = None) -> None: ...
    #   ic=None 时用 config 的 sane 默认（~0.5 附近 + 小随机）。
    #   禁止默认 hot IC。

    @property
    @abstractmethod
    def state(self) -> SubstrateState: ...

    @abstractmethod
    def perturb_history(self, delta: np.ndarray) -> None: ...
    #   Gate A 的 Lyapunov 计算需要：对整个 history 缓冲施加扰动。
```

**`kernels.py`**
```python
def dirac_delayed(history: np.ndarray, tau_steps: int) -> np.ndarray: ...
    # 返回 x(t - tau)，shape [k]

def powerlaw_delayed(history: np.ndarray, alpha: float, s0: float) -> np.ndarray: ...
    # 离散卷积 Σ_s K_pl(s)·x(t-s)，K_pl(s) ∝ (s+s0)^(-alpha-1)，权重归一化到 sum=1
    # 不变量：权重非负、和为 1

def hybrid_delayed(history, tau_steps, alpha, s0, w: float) -> np.ndarray: ...
    # w·dirac + (1-w)·powerlaw
```

**`dual_system.py` / `mackey_glass.py` — 主更新方程**
```
每一步（欧拉或 RK4，config 选）：
  K_i     = hybrid_delayed(history_i, τ_i, alpha, s0, w)
  dx_i/dt = β_i · K_i / (1 + K_i^n) − γ_i(y_i)·x_i + Σ_j W_ij·x_j + η_i + input_i
  dy_i/dt = ε_i · (|x_i| − y_i)
  γ_i     = γ_0 + κ·y_i
```
- `W`：耦合矩阵 [k,k]，config 提供（模拟 HPA 轴交叉调节）。Phase 0 固定。
- `η_i`：可选噪声，`noise_sigma`。**Gate A 跑 λ 时 noise_sigma=0**（混沌必须是内生的，不是噪声伪装）。
- 数值：默认 RK4，`dt` 与 `τ`（以步为单位）一致。`test_substrate.py` 必须验证长期有界（不 blow up）。

**不变量**：
- 自主运行（input=None, noise=0）时，状态既不塌到不动点（variance>阈值）也不发散（bounded）。这正是 Gate A 要量化的。

### 5.2 memory

**`reservoir.py` — Echo State Network (L1)**
```python
class Reservoir:
    def __init__(self, dim: int, spectral_radius: float, input_scaling, leak_rate, ...): ...
        # W_r: 稀疏随机，谱半径缩放到 spectral_radius < 1（echo state property）
        # 不变量：test_reservoir.py 断言 max|eig(W_r)| < 1

    def step(self, dt, x: np.ndarray, u: np.ndarray | None) -> np.ndarray: ...
        # dr/dt = -r/tau_r + tanh(W_r@r + W_x@x + W_in@u)
        # u = LLM 上一步输出 embedding 的投影（Phase 0 用合成信号）
        # 返回当前 r，shape [dim]

    @property
    def state(self) -> np.ndarray: ...   # r，shape [dim]
```
- `dim`：建议 128（v1 用 64，方差解释率只 31.5%）。是 config 参数。
- **验证**：`analysis/reservoir_probe.py` 喂入 5 组语义相似 vs 不相似的信号序列，断言 reservoir state 的余弦距离反映语义相似度。不反映 → reservoir 是 fancy noise generator，需重调。

**`hopfield.py`（Phase 3）**、**`slow_weights.py`（Phase 3）**：接口占位，Phase 0/1/2 不实现，只留 stub + docstring。触发用 z-score：`|x − running_baseline| / running_std > z_thresh`（默认 2.0），替代 v1 的固定 0.3。

### 5.3 llm

**`directions.py` — CAA 情感方向提取（离线）**
```python
def extract_directions(model, tokenizer, contrast_pairs, layers, method="caa") -> np.ndarray: ...
    # contrast_pairs: [(positive_prompt, negative_prompt), ...] 每种情感一组
    # 对每层取 (pos激活 − neg激活) 的均值，归一化
    # 返回 V，shape [k, d_model]（k 种情感方向）
    # 保存到 artifacts/directions.pt，供运行时加载
```
- 建议同时正交化（Gram-Schmidt）并记录方向间余弦相似度——v1 里焦虑/低落常高度共线。

**`hooks.py` — 残差流注入**
```python
class InjectionHook:
    # 用 PyTorch register_forward_hook 挂在 model.model.layers[i] 上（不依赖 TransformerLens）
    def set_delta(self, layer_idx: int, delta: torch.Tensor) -> None: ...
        # delta shape [d_model]，加到该层残差流输出（broadcast 到 seq 维）
    def clear(self) -> None: ...
```
- **首选原生 forward hook**，理由：对任意 HF 模型稳健，不受 TransformerLens 是否支持 Qwen 限制。nnsight 作为可选实现。

**`projection.py` — 状态 → steering 向量（核心）**
```python
@dataclass
class InjectionRecord:
    layer_norms_raw: dict[int, float]      # clip 前 per-layer ‖Δh‖
    layer_norms_clipped: dict[int, float]  # clip 后
    ratio_to_hidden: dict[int, float]      # ‖Δh‖ / ‖h‖
    clip_triggered: dict[int, bool]

def compute_injection(state: SubstrateState, r: np.ndarray, V, P, config) -> tuple[dict[int, Tensor], InjectionRecord]: ...
    # α_i = A_i · tanh(s_i · (x_i − x_baseline_i))            # 交感（来自 x）
    # β_i = B_i · tanh(s_i · (y_i − y_baseline_i))            # 副交感（来自 y）
    # Δh_L = P@r + Σ_i α_i·V_i − Σ_i β_i·V_i                  # 每个注入层 L
    #   （config 可选：副交感用不同方向 V_slow，让两系统语义各管一摊）
    # 记录每层 raw/clipped norm 到 InjectionRecord
    # cap 是 config，但主旋钮是 A_i：标定目标 = raw 95分位 < cap
```
维度契约（`test_projection.py` 必测）：`r∈[N]`, `P∈[d_model,N]`, `V∈[k,d_model]`, `α∈[k]`, `Δh∈[d_model]`。

### 5.4 feedback

**`sentiment.py`**
```python
class VADClassifier:
    def score(self, text: str) -> np.ndarray: ...
        # 返回 VAD [valence, arousal, dominance]，或直接映射到 k 维情感 external_input
        # 主实现：小型分类器（见 §9 候选模型）
        # fallback：中文情感词典（v1 方案，仅兜底）
```

**`loop.py`**
```python
class FeedbackLoop:
    def on_output(self, text: str) -> np.ndarray: ...
        # text → VAD → external_input（喂给 substrate.step）
    # ablation 开关：feedback_enabled: bool（Gate C 用）
```

### 5.5 analysis — 见 §6（gate 是它的主职责）

---

## 6. 验证门（本项目的心脏）

> **这是 v2 存在的理由。三道 gate 是硬性的。Codex 不得为通过而调低阈值——阈值即科学结论。**
> gate 结果统一写入 `artifacts/gate_results.json`，README 展示。

### Gate A — 自维持混沌（Phase 0，仅 substrate）

**问题**：substrate 无外部输入时，自己是否活着？还是塌成不动点（v1 T1）？

**脚本**：`scripts/gateA_lyapunov.py`
**方法**：Benettin 法算最大 Lyapunov 指数。
1. `noise_sigma=0`, `external_input=None`，sane IC，跑 transient（丢弃前 2000 步）。
2. 复制一条轨迹，用 `perturb_history()` 施加大小 `d0=1e-8` 的初始扰动（作用于整个 history 缓冲，因为延迟系统相空间是缓冲）。
3. 每 `renorm_interval` 步测参考轨迹与扰动轨迹的分离 `d_k`，累积 `log(d_k/d0)`，再把扰动重整化回 `d0`。
4. `λ_max = mean(log(d_k/d0)) / (renorm_interval · dt)`。

**PASS 条件（全部满足）**：
- `λ_max > 0.01`（稳健为正 → 混沌，对冲 T1）
- 状态有界：`max|x| < blowup_threshold`（不发散）
- 状态不塌缩：末段 `var(x) > collapse_threshold`（不是不动点）

**输出**：`λ_max`、每维 range、自相关函数、`x(t)` 波形图。

**若 FAIL**：先跑 `scripts/sweep_kernel.py` 扫 `w∈[0,1]` × `alpha`，定位混沌窗口。若 MG+混合核在所有参数下都 FAIL → config 切 `core: hindmarsh_rose`，重跑 Gate A。**这一步没过，不许碰 Phase 1。**

### Gate B — 输入敏感性（Phase 0，substrate + 合成输入）

**问题**：substrate 对输入*内容*敏感，还是所有"演化"都只是 IC 回弹（v1 T3）？

**脚本**：`scripts/gateB_sensitivity.py`
**方法**：**相同 IC、相同 seed**，两条运行：
- Run S：注入合成"强情感" `external_input` 序列（大幅度 VAD）
- Run N：注入合成"中性"序列（近零幅度）
- 第三条 baseline B：两次都用中性，测纯噪声导致的轨迹差异 `d_noise`

**PASS 条件**：`dist(x_S(t), x_N(t)) > 5 · d_noise`（强情感明显改变轨迹，效应量可测）。

**输出**：两条 `x(t)` 叠加图、轨迹距离时间序列、效应量。

**副产物**：顺带跑 `reservoir_probe.py`，验证 reservoir 编码语义（D5）。

### Gate C — 反馈因果性（Phase 2，需真实闭环）

> harness 在 Phase 0 就用合成信号搭好（`analysis/ablation.py`），但**判定 PASS 用真实 LLM 闭环**。

**问题**：VAD 反馈回路真的闭合了吗？还是 feedback 对轨迹贡献 ≈0（我赌 v1 的贡献 <20%）？

**脚本**：`scripts/gateC_causality.py`
**方法**：相同 user prompt 序列，两条件：
- 条件 Full：substrate 状态调制注入 → 塑造输出 → VAD → 反馈回 substrate（完整闭环）
- 条件 Ablated：注入冻结/置零（输出不被 substrate 塑造），substrate 仍运行并接收（未被塑造的）输出的 VAD

**PASS 条件**：两条件产生**可测差异**——substrate 轨迹不同 *且* 输出分布不同（如输出 embedding 分布距离、或人工/自动情感标注差异显著）。若两者相同 → 回路是开的，feedback 无因果贡献，需修。

**输出**：两条件 `x(t)` 对比、输出分布距离、消融贡献占比。

---

## 7. 分阶段实施计划

> Codex：**顺序执行，每 Phase 有明确 exit criteria。前一 Phase 未达标不进入下一 Phase。**

### Phase 0 — Substrate 验证（无 LLM）★ 最关键
**建**：`substrate/*`, `memory/reservoir.py`, `analysis/{lyapunov,sensitivity,reservoir_probe,viz}.py`, 相关 tests。
**做**：
1. 实现 Mackey-Glass + 混合核 + 双系统 + reservoir。
2. `run_bare_substrate.py` 裸跑可视化（波形、相图、γ_eff）。
3. 跑 **Gate A**。FAIL → sweep_kernel → 必要时切 Hindmarsh-Rose，直到 PASS。
4. 跑 **Gate B**（合成输入）。
5. reservoir 语义编码验证。
**Exit**：Gate A + Gate B 均 PASS 并写入 `gate_results.json`。**否则停在这里。**

### Phase 1 — LLM 注入管线
**建**：`llm/{directions,hooks,projection}.py`, `scripts/{extract_directions,calibrate_injection}.py`, `test_projection.py`。
**做**：
1. CAA 提取情感方向 V，记录方向间余弦相似度。
2. forward hook 注入 + `compute_injection` + `InjectionRecord`。
3. **`calibrate_injection.py`**：扫 `A_i`，输出 per-layer raw norm 直方图，标定到 95分位 < cap（对冲 T2）。
4. 验证：**相同 prompt + 不同 substrate 状态 → 不同输出**（确定性解码 `do_sample=False`，排除采样噪声）。
**Exit**：注入标定完成（raw norm 直方图达标）；同 prompt 对 substrate 状态敏感。

### Phase 2 — 闭合真实回路
**建**：`feedback/{sentiment,loop}.py`, `llm/server.py`, `scripts/{gateC_causality,run_full_system}.py`。
**做**：
1. VAD 分类器（替代词典）。
2. 完整闭环 + HTTP server（端点见 §8）。
3. 跑 **Gate C**（真实闭环）。
**Exit**：Gate C PASS（feedback 有可测因果贡献）。

### Phase 3 — 情节 + 慢记忆（进阶，可选）
**建**：`memory/{hopfield,slow_weights}.py`。
**做**：z-score 触发的 Hopfield 情节记忆；Hebbian 慢权重塑形 W_r/V。谨慎加衰减约束防漂移。
**Exit**：无硬 gate；验证联想回忆（相似状态自发触发存储片段并影响 r）。

---

## 8. 配置 schema

`config/schema.py` 用 pydantic 定义并校验。`config/default.yaml` 给全量默认，`config/phase0.yaml` 专门调混沌。关键字段：

```yaml
substrate:
  core: mackey_glass            # mackey_glass | hindmarsh_rose  ← Gate A 兜底开关
  k: 4                          # 情感维度数
  dim_names: [arousal, calm, positive, negative]
  integrator: rk4               # rk4 | euler
  dt: 0.1
  noise_sigma: 0.0              # Gate A 必须 0
  ic_mode: sane                 # sane（~0.5）| custom；禁止 hot 默认
  mackey_glass:
    beta: [...]                 # 每维
    gamma_0: 1.0
    kappa: 0.5
    n: 10
    tau_steps: [...]            # 每维延迟（步）
    W: [[...]]                  # 耦合矩阵 [k,k]
  slow:
    eps: [...]                  # ε_i ≪ 1/τ_i
  kernel:
    type: hybrid                # dirac | powerlaw | hybrid
    w: 0.7                      # 混合权重（sweep 目标）
    alpha: 0.3                  # 幂律指数（sweep 目标）
    s0: 1.0
    buffer_len: 500

reservoir:
  dim: 128                      # v1 用 64 偏小
  spectral_radius: 0.95
  leak_rate: 0.3
  tau_r: 5.0
  input_scaling: 0.1

llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # ← 确认精确 HF repo id（v1 写 Qwen3.5-9B，需核对）
  dtype: bfloat16
  inject_layers: [15, 18, 21, 24]
  deterministic: true           # do_sample=False，排除采样噪声
  injection:
    A: [...]                    # 每维交感注入幅度（主标定旋钮）
    B: [...]                    # 每维副交感幅度
    s: [...]                    # tanh 斜率
    x_baseline: [...]
    y_baseline: [...]
    cap: 8.0                    # clip 上限（次要，靠 A 让它几乎不触发）
    slow_use_separate_directions: false

feedback:
  enabled: true                 # Gate C ablation 开关
  classifier: vad               # vad | lexicon(fallback)
  steps_per_turn: 50            # 每 turn 推进多少 substrate 步

gates:
  lyapunov:  { lambda_min: 0.01, blowup_threshold: 100, collapse_threshold: 0.001, transient: 2000, renorm_interval: 20, d0: 1e-8 }
  sensitivity: { noise_multiple: 5 }
```

**Server 端点**（Phase 2，沿用 v1）：`/generate` · `/feedback` · `/reset` · `/memory` · `/health`。

---

## 9. 依赖与环境

**环境**：Windows 工作站，RTX 5090（32GB）。VRAM 预算 ~22GB（v1 实测）。

`pyproject.toml` 锁版本（Codex：安装后 `pip freeze` 存 `requirements.lock`）：
```
python = ">=3.11,<3.13"
torch          # 与 5090/CUDA 匹配的版本（sm_120）
transformers
accelerate
numpy
scipy          # Lyapunov、信号处理
matplotlib
pyyaml
pydantic>=2
fastapi + uvicorn   # server
nnsight        # 可选，hook 备选方案
```
**情感分类器候选**（择一，Phase 2 定）：中文情感/VAD 微调模型（如基于 `chinese-roberta-wwm-ext` 微调 valence/arousal），或多语言 sentiment 模型兜底；词典法仅 fallback。

**注意**：RTX 5090 是 sm_120，需较新的 PyTorch + CUDA。若 pip 装的 torch 不支持，Codex 应在 README 记录需要的 nightly/特定 wheel，并提示用户网络设置可能需放行相应源。

---

## 10. 给 Codex 的执行须知

1. **严守 Phase 顺序与 gate**。Phase 0 的 Gate A/B 未 PASS，不写任何 `llm/` 代码。
2. **不许为过 gate 调阈值**。阈值是科学结论。过不了就修系统（换核、扫参数），不是改数字。
3. **每个模块配 test**，先写契约测试（维度、不变量）再写实现。
4. **确定性解码**：Phase 1 起所有 LLM 调用 `do_sample=False`，否则无法区分"substrate 影响"和"采样噪声"。
5. **记录一切注入**：`compute_injection` 必返 `InjectionRecord`，别偷懒省掉——T2 就是这么来的。
6. **Sane IC**：任何 `reset()` 默认不许 hot IC。
7. **中文可视化字体**：matplotlib 配 `plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']`（或系统可用 CJK 字体），修 v1 图1 的乱码。
8. **核对模型名**：`Qwen3.5-9B` 需确认精确 HF repo id 再写进 config。
9. **gate 结果落盘**：`artifacts/gate_results.json` + README 表格，每次跑 gate 更新。

---

## 附录 A · 关键公式汇总

```
混合延迟核:   K_i = w·x_i(t−τ_i) + (1−w)·Σ_s K_pl(s)·x_i(t−s),  K_pl(s) ∝ (s+s0)^(−α−1)
快系统:       dx_i/dt = β_i·K_i/(1+K_i^n) − γ_i(y_i)·x_i + Σ_j W_ij·x_j + η_i + input_i
慢系统:       dy_i/dt = ε_i·(|x_i| − y_i)
自适应阻尼:   γ_i(y_i) = γ_0 + κ·y_i
储备池:       dr/dt = −r/τ_r + tanh(W_r·r + W_x·x + W_in·u)
注入(交感):   α_i = A_i·tanh(s_i·(x_i − x̄_i))
注入(副交感): β_i = B_i·tanh(s_i·(y_i − ȳ_i))
残差注入:     h'_L = h_L + P·r + Σ_i α_i·V_i − Σ_i β_i·V_i
Lyapunov:     λ_max = ⟨log(d_k/d0)⟩ / (renorm_interval·dt)
Hopfield触发: store when |x − baseline|/σ_running > z_thresh
Hopfield检索: recall = Σ_k softmax(ξ·⟨r, r_k⟩)_k · (r_k, x_k, y_k)
```

## 附录 B · 调参 troubleshooting

| 症状 | 可能原因 | 处理 |
|---|---|---|
| Gate A: λ ≤ 0，塌到不动点 | 幂律权重过大 / w 太小 / 阻尼太强 | 提高 w；降 α；降 γ_0；`sweep_kernel.py` 找窗口 |
| Gate A: 发散 blow up | β 太大 / n 太小 / dt 太大 | 降 β；提 n；减 dt；改 RK4 |
| Gate A 全参数 FAIL | MG+混合核几何不兼容长记忆 | 切 `core: hindmarsh_rose` |
| Gate B: 轨迹不分化 | input_scaling 太小 / 注入被阻尼吃掉 | 提 input 幅度；查 external_input 是否真进了方程 |
| reservoir probe 不反映语义 | 谱半径太小/太大 / dim 不够 | 调 spectral_radius 近 0.95；dim→128/256 |
| 注入 raw norm 常顶 cap | A_i 太大 | 降 A_i 到 95分位 < cap；别只提 cap |
| 同 prompt 输出不变 | 注入太弱 / 层选得不对 | 提 A_i；换中后层；查 hook 是否真挂上 |
| Gate C: 两条件无差异 | 回路没接通 / VAD 不敏感 | 查 feedback→input 通路；换更敏感分类器 |
| 输出胡言乱语 | ‖Δh‖/‖h‖ > 0.2 | 降注入幅度，监控 ratio_to_hidden |

---

*v2 的成功标准不是"看起来像情感"，而是三道 gate 全绿：substrate 自己活着（A）、对内容敏感（B）、回路真闭合（C）。绿了，你就有一个真正在呼吸的 substrate。*
