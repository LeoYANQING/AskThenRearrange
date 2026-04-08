# AskThenRearrange：系统技术概览

> **文档状态**：反映当前代码实现（截至 2026-04-03）。实验验证进行中。
> 理论框架详见 `unified_question_planning.md`；Entropy 方法推导详见 `entropy_method.md`。

---

## 1. 系统目标

在有限的提问预算 $B$ 内，通过与用户的多轮对话推断所有物品的正确摆放位置。

**输入**：房间名、收纳位置列表 $\mathcal{C}$、已展示物品 $\mathcal{O}_s$、未展示物品 $\mathcal{O}_u$、预算 $B$  
**输出**：每个物品的预测摆放位置 $\hat{c}(o)$  
**评价指标**：seen accuracy（可见物品正确率）、unseen accuracy（不可见物品正确率）

---

## 2. 三种提问模式（Question Patterns）

| 模式 | 缩写 | 语义 | 单次覆盖范围 |
|---|---|---|---|
| Action-Oriented | AO | 直接询问某个物品的摆放位置 | 1 个 seen object |
| Preference Eliciting | PE | 询问某类物品的高层偏好规则 | 多个 seen + 泛化到 unseen |
| Preference Induction | PI | 基于已确认行为归纳规则并请用户确认 | 多个 seen + 泛化到 unseen |

**关键区别**：
- AO 获取确定性的点位信息，零泛化；
- PE 主动探索用户偏好（无需已有证据），覆盖范围广但答案可能模糊；
- PI 被动归纳已有行为证据，需要≥2条 confirmed_actions 才能触发，答案通常更清晰。

---

## 3. 五种策略模式（Policy Modes）

策略模式定义了在给定状态下**允许使用哪些 pattern**，以及**选择逻辑**。

### 3.1 Direct Querying（直接查询）

```
允许 pattern：AO only
逻辑：始终选 AO
```

用户假设：用户对每个物品的具体位置都清楚，可直接逐一询问。

---

### 3.2 User-Preference First（偏好优先）

```
允许 pattern：PE、AO
逻辑：只要有 ≥2 个 unresolved objects 未被 confirmed_preferences 覆盖，就选 PE；否则选 AO
```

用户假设：用户可以清晰表达抽象偏好规则，且这些规则能够覆盖多个物品。

**优化 U3（2026-04-03）**：PE proposer 强制输出 receptacle 最佳猜测，产出"Do [category] usually end up in [receptacle]?"式确认问题，而非开放式询问。

---

### 3.3 Parallel Exploration（并行探索）

```
允许 pattern：AO、PI（无 PE）
逻辑：
  ├── 若 unsummarized confirmed_actions ≥ 2 → PI（定向指导：说明主导 receptacle 和示例物品）
  └── 否则 → AO（集群导向：引导选与已有动作同一 receptacle 的物品）
```

用户假设：用户不善于直接表达抽象偏好，但可以从具体行为中归纳规则。

**设计要点（优化 P1/P2/P3，2026-04-03）**：
- 阈值从 3 降至 **2**，使 PI 在 budget=3 时可以触发（AO×2 → PI×1）
- **集群导向 AO**：第一条 AO 确认后，后续 AO guidance 显式指向同一 receptacle，使证据集中
- **定向 PI guidance**：PI 触发时 guidance 明确写出"'book' 和 'novel' 都去了 'bookshelf'，确认这是通用规则"，提高 PI proposer 的问题精度

---

### 3.4 Hybrid All（混合策略）

```
允许 pattern：PE、PI、AO
逻辑（LLM 驱动）：
  ├── 优先 PE（若 uncovered ≥ 2）
  ├── 次之 PI（若 unsummarized ≥ 3）
  └── 否则 AO
```

与 UPF 的关系：在 12 个 seen objects、budget ≤ 5 的条件下，uncovered 始终 ≥ 2，PE 始终获选。因此 **Hybrid All ≡ User-Preference First**（实验已多次复现）。这是一个有意义的结构性发现：PE 在存在时会完全主导混合策略。

---

### 3.5 PI Cold-Start（冷启动 PI）

```
允许 pattern：PI（优先）、AO（备用）
逻辑：只要有 ≥2 confirmed_actions 就立即切换 PI；否则 AO
```

实验结论（10 episodes，budget=[1,3,5]）：
- B=3 时 seen_accuracy = 0.400，与 B=1 相同，PI 无提升
- B=5 时 0.600，低于 Direct Querying（0.617）
- **结论**：预算紧张时 1 次 PI 无法补偿 1 次 AO 的损失；PE 的效率显著高于 PI

---

## 4. 核心组件与代码映射

```
单轮决策流程：
AgentState(S_t)
  │
  ├─ QuestionPolicyController.plan_next_question(mode)
  │    ├─ _allowed_patterns()       → 确定可用 pattern
  │    ├─ _rule_*(state)            → 规则选 pattern + 生成 guidance
  │    └─ [可选] _entropy_select()  → EER 选 pattern（selection_method="entropy"）
  │
  ├─ {Action,Eliciting,Induction}Proposer.propose(state, guidance)
  │    → 生成具体问题文本（intent.question）
  │
  ├─ NaturalUserOracle.answer(question)
  │    → 模拟用户回答
  │
  ├─ StateUpdate.update_state_from_*_answer(state, answer)
  │    → 解析回答，更新 confirmed_actions / confirmed_preferences
  │
  └─ recompute_online_placements(state)
       → 从 unresolved_objects 移除已确认物品
```

| 概念 | 文件 | 核心类/函数 |
|---|---|---|
| 策略控制器 | `question_policy.py` | `QuestionPolicyController` |
| 三种 proposer | `proposers.py` | `ActionProposer`, `PreferenceElicitingProposer`, `PreferenceInductionProposer` |
| 状态更新 | `state_update.py` | `StateUpdate` |
| 智能体状态 | `agent_schema.py` | `AgentState` TypedDict |
| 置信度估计（Entropy 路径）| `belief_estimator.py` | `BeliefEstimator` |
| 最终放置预测 | `evaluation.py` | `FinalPlacementPlanner` |
| 实验循环 | `test_policy_loop.py` | `run_policy_loop`, `run_ablation_experiment` |
| 并行实验 | `parallel_runner.py` | `run_parallel_ablation` |

---

## 5. 统一选择机制：EER（Expected Entropy Reduction）

**核心设计**（2026-04-03 统一）：所有 multi-pattern 模式使用 **同一个 EER 机制** 选择 pattern。

```
plan_next_question(state, mode):
    allowed = _allowed_patterns(state, mode)
    if len(allowed) == 1:    →  直接返回（DQ 走这条路）
    if len(allowed) >= 2:    →  EER 选择（UPF/Parallel/Hybrid 走这条路）
```

**EER 计算公式**：

```
EER_action    = H(max_entropy_object)
EER_eliciting = max_group( Σ H(obj) + α_elic × n_unseen × H_max )    α_elic = 0.5
EER_induction = Σ H(matched) + α_ind × n_unseen_matched × H_max       α_ind  = 1.0
```

**α 系数的设计逻辑**：
- PE（α=0.5）是**推测性**的——不知道用户是否接受这个类别假设
- PI（α=1.0）是**证据驱动**的——已有 confirmed_actions 证明模式存在
- 因此 PI 对 unseen 泛化的权重是 PE 的 **2 倍**，这是 Hybrid 区分于 UPF 的关键机制

**各 mode 在 EER 下的行为预测**：

| Mode | 允许 patterns | EER 行为 |
|---|---|---|
| DQ | {AO} | 单 pattern → 跳过 EER → 逐个物品提问 |
| UPF | {PE, AO} | 早期 PE 占优（分组大）→ 后期 AO 接管 |
| Parallel | {AO, PI} | AO 积累证据 → PI 在证据聚集后触发 |
| Hybrid | {PE, PI, AO} | PE 早期探索 → PI 锁定规则（α=1.0 > α=0.5）→ AO 清理 |

**Hybrid 与 UPF 的分化机制**：

PE 轮产出 confirmed_actions → PI 变为可用 → EER_PI 的 α=1.0 使其可能超过 EER_PE（α=0.5）：

```
Turn 1:  allowed = {PE, AO}           → EER 选 PE → 确认3个物品 → 3 confirmed_actions
Turn 2:  allowed = {PE, PI, AO}       → PI 可用！
         EER_PE ≈ 4×1.5 + 0.5×2×2.5 = 8.5     （新组，推测性）
         EER_PI ≈ 2×1.5 + 1.0×3×2.5 = 10.5    （已有证据，高泛化权重）
         → EER 选 PI → 锁定规则，覆盖更多 unseen
Turn 3+: allowed = {PE, PI, AO}       → 继续最优选择
```

UPF 没有 PI → 无法利用 α=1.0 的证据泛化优势。

---

## 6. 关键工程优化（2026-04-03）

### 6.1 parallel_exploration 的三项修复

**背景**：实验发现 parallel_exploration 持续低于 direct_querying（B=3: 0.475 vs 0.508）。

| 问题 | 根因 | 修复 |
|---|---|---|
| PI 从未触发（B=3） | 阈值为 3，budget=3 时 AO 消耗全部预算 | 阈值降至 **2**（`_induction_is_available` 已支持） |
| PI 归纳无效 | AO 问不同类别物品 → 3条动作分属3个 receptacle → PI 无可归纳模式 | **集群导向 AO guidance**：每次 AO 后提示下一轮选同一 receptacle 的物品 |
| PI 问题太泛 | PI guidance 是通用文本，proposer 生成模糊归纳问题 | **定向 PI guidance**：自动生成含主导 receptacle + 示例物品名的具体提示 |

### 6.2 user_preference_first 的三项修复

**背景**：PE 回答中 receptacle 提取失败 → 物品仅进入 confirmed_preferences（无 receptacle）而非 confirmed_actions → 最终预测仍靠 planner 猜测。

| 问题 | 根因 | 修复 |
|---|---|---|
| receptacle 提取失败 | state_update LLM 未填 `category_rule_receptacle`，且只扫描 category_rule 文本 | **扫描 oracle 原始回答**：`_fuzzy_match_receptacle(answer, receptacles)` 作为最后回退 |
| PE 覆盖物品太少 | prompt 只说"covered_objects 的子集"，LLM 只返回锚点示例 | **广泛覆盖 prompt**："扫描全部 seen_objects，找出所有类似物品，不限于锚点" |
| PE 问题模糊 | receptacle 为 null 时 proposer 生成开放式问题，oracle 回答难以解析 | **强制 receptacle 猜测**：将字段从"可选"改为"强烈建议填写最佳猜测" |

---

## 7. 实验结果追踪

### 7.1 pre-optimization 基线（10 episodes, budget=[1,3,5], rule-based）

| Mode | B=1 Seen/Unseen | B=3 Seen/Unseen | B=5 Seen/Unseen |
|---|---|---|---|
| direct_querying | 0.392 / 0.492 | 0.508 / 0.525 | 0.617 / 0.608 |
| user_preference_first | 0.458 / 0.450 | 0.683 / 0.608 | 0.767 / 0.683 |
| parallel_exploration | 0.400 / 0.483 | 0.475 / 0.492 | 0.567 / 0.533 |
| hybrid_all | 0.450 / 0.467 | 0.683 / 0.608 | 0.767 / 0.683 |
| pi_cold_start | 0.400 / 0.475 | 0.400 / 0.467 | 0.600 / 0.542 |

**关键发现**：
1. UPF ≡ Hybrid（结构性等价，B=3 和 B=5 完全相同）
2. Parallel < Direct（PI 从未在 B=3 触发，B=5 仅触发一次且证据分散）
3. PI Cold-Start 在 B=3 seen_accuracy 与 B=1 相同（PI 无提升）

### 7.2 post-optimization（进行中）

目标：对比优化前后的 user_preference_first、parallel_exploration、direct_querying。  
命令：`parallel_runner.py --modes user_preference_first,parallel_exploration,direct_querying --num-samples 10 --budget-list 1,3,5`

---

## 8. 并行实验基础设施

### 8.1 GPU 利用率优化

| 配置 | 速度 | GPU 利用率 |
|---|---|---|
| test_policy_loop.py（串行） | 1x 基准 | ~30–40% |
| parallel_runner.py（3 worker） | **~2.8x** | ~85–95% |

- 三个 Ollama 实例分别运行在端口 11435 / 11436 / 11437，每个实例独占 2 块 GPU（tensor 并行）
- episode-level 并行：每个线程持有独立的 LLM 实例，通过 Queue 分配 worker URL
- log 格式与 test_policy_loop.py 完全兼容（相同的 `episode_finished` / `budget_aggregated` 事件结构）

### 8.2 运行命令参考

```bash
# 指定 mode 的快速验证（推荐）
python3 parallel_runner.py \
  --modes user_preference_first,parallel_exploration,direct_querying \
  --num-samples 10 --budget-list 1,3,5 \
  --ablation-log logs/validation_opt_10ep.jsonl \
  --output plots/validation_opt_10ep.png

# 全量实验（102 episodes × 5 modes × budget=[1,3,5]）
python3 parallel_runner.py \
  --num-samples 102 --budget-list 1,2,3,4,5,6 \
  --ablation-log logs/full_ablation.jsonl \
  --output plots/full_ablation.png
```

---

## 9. 论文定位与下一步

### 投稿目标：IJHCS（B方向）
- **核心贡献**：不同主动提问策略（Policy Mode）在客观指标（accuracy）和主观指标（UX）上的系统性对比
- **Design Implication**：何时用哪种策略、各策略的适用假设

### Entropy 方法的定位
当前决策：Entropy（EER）作为**辅助分析工具**而非核心系统组件：
- Section 3（Methods）：介绍规则驱动的策略框架，Entropy 作为理论基础说明为何某种规则更优
- Discussion：用 EER 计算解释实验差异（如 UPF 为何在 B=3 显著优于 Direct）
- 若审稿人质疑规则设计的 ad hoc 性，可用 Entropy 数值说明规则与信息论最优选择的一致性

### 待完成
- [ ] post-optimization 实验结果分析（bodjyuk89 进行中）
- [ ] Study 1 最终表格：4 modes × budget=[1,3,5,6] × 102 episodes
- [ ] Study 2 用户研究设计
- [ ] Methods 章节写作（基于本文档）
