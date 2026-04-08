# 统一提问规划框架：基于期望熵减的联合模式-目标选择

> **写作状态**：理论草稿（v1）。本文件用于在实现前厘清框架设计，后续将整合进论文正式 Methods 章节。

---

## 3.1 问题形式化

给定一个家居整理场景，智能体需要在有限的提问预算 $B$ 内，通过与用户的多轮对话推断出所有物品的正确摆放位置。场景由元组 $(R, \mathcal{C}, \mathcal{O}_s, \mathcal{O}_u)$ 定义，其中 $R$ 为房间名，$\mathcal{C} = \{c_1, \dots, c_M\}$ 为可用收纳位置集合，$\mathcal{O}_s$ 和 $\mathcal{O}_u$ 分别为已展示物品（seen）与未展示物品（unseen）。

智能体的目标是最大化任务完成率：

$$\max_{\{q_1, \dots, q_B\}} \frac{1}{|\mathcal{O}_s \cup \mathcal{O}_u|} \sum_{o} \mathbf{1}[\hat{c}(o) = c^*(o)]$$

其中 $c^*(o)$ 为用户的真实偏好位置，$\hat{c}(o)$ 为智能体的最终预测。

---

## 3.2 智能体状态表示

我们维护结构化的智能体状态 $\mathcal{S}_t$，在第 $t$ 轮对话后包含：

- **已确认放置** $\mathcal{A}^+ = \{(o_i, c_j)\}$：经用户确认的物品-位置对
- **已确认偏好** $\mathcal{P}^+ = \{(h_k, \mathcal{O}_k, c_k)\}$：用户偏好规则，$h_k$ 为自然语言描述，$\mathcal{O}_k$ 为覆盖物品集，$c_k$ 为对应位置
- **否定证据** $\mathcal{A}^-$，$\mathcal{P}^-$：被否定的放置与偏好
- **未解决物品** $\mathcal{U}_t \subseteq \mathcal{O}_s$：尚未确定摆放位置的可见物品
- **问答历史** $\mathcal{Q}_t = \{(q_i, a_i)\}_{i=1}^{t}$：已进行的对话记录

---

## 3.3 三种提问模式

我们定义三种互补的提问模式（question patterns），每种模式获取不同粒度的用户信息：

**Action-Oriented（直接提问）**：针对单个未解决物品 $o \in \mathcal{U}_t$ 直接询问其摆放位置，获取确定性的物品-位置映射。信息增益集中但覆盖范围有限，单次提问至多解决一个物品的不确定性。

**Preference Eliciting（偏好引导）**：询问用户关于某一类别物品的高层偏好规则（如"电子设备通常放在哪里？"）。一次回答可同时覆盖多个物品的放置决策，也可泛化到未展示物品，具有较高的信息效率。

**Preference Induction（偏好归纳）**：基于已收集的行为证据 $\mathcal{A}^+$，向用户提出归纳性假设（如"我注意到您把书籍都放在书架上，这是通用规则吗？"），以确认或修正从观察中提炼的偏好模式。该模式可将已有证据泛化至更多物品，但要求已有足够的行为基础。

---

## 3.4 提问规划的形式化表述

### 3.4.1 提问空间

**核心洞察**：一个"问题"不仅由其模式（pattern）决定，还由其作用目标（target）决定。我们将提问空间形式化为一个二元组集合：

$$\mathcal{Q} = \{(\text{pattern}, \text{target})\}$$

其中不同模式对应不同类型的 target：

| 模式 | Target 类型 | 语义 |
|------|------------|------|
| action | 单个物品 $o \in \mathcal{U}_t$ | 询问该物品的具体位置 |
| eliciting | 物品分组 $G_c \subseteq \mathcal{U}_t$ | 询问分组共享的类别偏好 |
| induction | 行为模式 $G_p \subseteq \mathcal{U}_t$ | 确认从 $\mathcal{A}^+$ 归纳出的偏好规则 |

### 3.4.2 贪心规划原则

多步提问规划在理论上是一个部分可观测马尔可夫决策过程（POMDP），需要在信念状态空间上做值函数规划。在预算有限（$B \leq 6$）和状态空间巨大的现实条件下，我们采用**单步贪心近似**：

$$q_t^* = \arg\max_{q \in \mathcal{Q}_{\text{valid}}(\mathcal{S}_t)} \text{EER}(q \mid \mathcal{S}_t)$$

其中 $\text{EER}(q \mid \mathcal{S}_t)$ 为期望熵减（Expected Entropy Reduction），衡量提问 $q$ 在当前状态下的期望信息收益；$\mathcal{Q}_{\text{valid}}(\mathcal{S}_t)$ 为当前策略模式允许的有效提问集合（见 3.8 节）。

### 3.4.3 与现有方法的对比

| | 现有方法（两层独立决策） | 本框架（联合选择） |
|--|--|--|
| Pattern 选择 | EER（pattern 级）或规则 | EER（(pattern, target) 联合级）|
| Target 选择 | LLM 自由生成 | EER 最大化 |
| Policy mode 表达 | if/else 规则 | 有效集合约束 |
| 端到端优化 | 否 | 是 |

---

## 3.5 基于 LLM 的置信度估计

不确定性驱动策略的核心在于对每个未解决物品的放置不确定性进行量化。我们利用大语言模型作为**置信度估计器**（Belief Estimator），在每轮决策前评估所有相关物品的摆放可能性。

给定当前状态 $\mathcal{S}_t$，LLM 为每个物品 $o$ 生成 Top-3 最可能收纳位置及其概率估计 $\{(c_k, p_k)\}_{k=1}^{3}$，其中 $\sum_{k=1}^{3} p_k \leq 1$。剩余概率质量均匀分配至其他 $M-3$ 个位置，从而得到完整的置信度分布：

$$\hat{P}(c \mid o, \mathcal{S}_t) = \begin{cases} p_k & \text{if } c = c_k \text{ for some } k \leq 3 \\ \frac{1 - \sum_k p_k}{M - 3} & \text{otherwise} \end{cases}$$

**对 unseen 物品的估计**：当 EER 计算需要估计未展示物品 $o \in \mathcal{O}_u$ 的分布时，BeliefEstimator 在同一调用中将其包含在查询物品列表中。LLM 可利用已知偏好规则 $\mathcal{P}^+$ 对 unseen 物品做合理的置信度推断（例如，若 $\mathcal{P}^+$ 中包含"书籍→书架"规则，则未展示的书籍应获得较高的书架置信度）。

**概率校准**：为缓解 LLM 输出模板化概率（如所有物品均输出 0.40/0.30/0.20）的问题，我们在提示中引入显式校准指引：
- **OBVIOUS**（常识性明确放置）：Top-1 概率在 0.70–0.90
- **LIKELY**（有证据支撑）：Top-1 概率在 0.45–0.65
- **UNCERTAIN**（无充分信息）：概率均匀分散，Top-1 在 0.25–0.35

实验表明该策略可将熵值范围从 [0.000, 0.400] 扩展到 [0.000, 1.528]，显著提高了信息量的区分度。

---

## 3.6 Shannon 熵与目标分组

### 3.6.1 物品级不确定性

基于置信度分布，每个物品 $o$ 的放置不确定性由 Shannon 熵量化：

$$H(o \mid \mathcal{S}_t) = -\sum_{c \in \mathcal{C}} \hat{P}(c \mid o, \mathcal{S}_t) \log_2 \hat{P}(c \mid o, \mathcal{S}_t)$$

最大熵为 $H_{\max} = \log_2 M$（完全均匀分布），最小熵为 0（完全确定）。

### 3.6.2 目标分组

为支持 eliciting 和 induction 的 EER 计算，我们从置信度分布和状态中提取两类分组结构：

**Eliciting 分组**：基于 Top-1 预测位置对未解决物品进行分组——共享同一预测位置的物品可能被同一条类别偏好规则覆盖：

$$G_c^{\text{elic}} = \{o \in \mathcal{U}_t : \hat{c}_1(o) = c\}, \quad \hat{c}_1(o) = \arg\max_c \hat{P}(c \mid o, \mathcal{S}_t)$$

仅考虑规模 $|G_c^{\text{elic}}| \geq 2$ 的分组（单个物品不构成类别规则的查询基础）。

**Induction 分组**：基于 $\mathcal{A}^+$ 中未归纳的确认放置，提取主导收纳位置 $c_d$（出现频率最高）并找到预测与之匹配的未解决物品：

$$c_d = \arg\max_c \left|\{a \in \mathcal{A}^+_{\text{unsummarized}} : a.\text{receptacle} = c\}\right|$$

$$G_{c_d}^{\text{ind}} = \{o \in \mathcal{U}_t : \hat{c}_1(o) = c_d\}$$

### 3.6.3 泛化奖励

偏好类问题（eliciting / induction）的一个关键优势是其答案可以**泛化**至未展示物品 $\mathcal{O}_u$。我们通过泛化奖励（Generalization Bonus）将这一隐性收益显式纳入 EER 计算：

$$\text{Bonus}(\text{pattern}, \text{group}) = \alpha_{\text{pattern}} \cdot n_{\text{unseen}}(\text{group}) \cdot H_{\max}$$

其中 $n_{\text{unseen}}(\text{group})$ 为可能受该偏好规则覆盖的 unseen 物品数量，$\alpha_{\text{pattern}}$ 为可靠性系数：

- $\alpha_{\text{eliciting}} = 0.5$：类别偏好规则约有 50% 的概率适用于同类 unseen 物品
- $\alpha_{\text{induction}} = 1.0$：从行为证据归纳出的规则可靠性更高，按完全适用计

---

## 3.7 联合期望熵减（Joint EER）

**期望熵减**（EER）为问题 $q = (\text{pattern}, \text{target})$ 的统一信息价值度量：

$$\text{EER}(q \mid \mathcal{S}_t) = \mathbb{E}_{a \sim P(a \mid q, \mathcal{S}_t)}\left[\sum_{o \in \mathcal{U}_t} \left(H(o \mid \mathcal{S}_t) - H(o \mid \mathcal{S}_{t+1})\right)\right]$$

在近似条件下（假设一次回答可完全解决对应分组内的不确定性），EER 对三类提问有如下封闭形式：

### Action-Oriented EER

直接提问可完全解决目标物品 $o$ 的不确定性，$H(o \mid \mathcal{S}_{t+1}) = 0$：

$$\text{EER}_{\text{action}}(o) = H(o \mid \mathcal{S}_t)$$

最优 action 目标为当前熵最高的物品：$o^* = \arg\max_{o \in \mathcal{U}_t} H(o)$。

### Preference Eliciting EER

偏好引导问题作用于分组 $G_c^{\text{elic}}$，成功时可解决分组内所有物品及部分 unseen 物品的不确定性：

$$\text{EER}_{\text{eliciting}}(G_c^{\text{elic}}) = \sum_{o \in G_c^{\text{elic}}} H(o \mid \mathcal{S}_t) + \alpha_{\text{eliciting}} \cdot n_{\text{unseen}}(c) \cdot H_{\max}$$

最优 eliciting 目标为 EER 最大的分组：$c^* = \arg\max_{c} \text{EER}_{\text{eliciting}}(G_c^{\text{elic}})$，且 $|G_c^{\text{elic}}| \geq 2$。

### Preference Induction EER

偏好归纳作用于主导模式分组 $G_{c_d}^{\text{ind}}$，基于已有行为证据泛化：

$$\text{EER}_{\text{induction}}(G_{c_d}^{\text{ind}}) = \sum_{o \in G_{c_d}^{\text{ind}}} H(o \mid \mathcal{S}_t) + \alpha_{\text{induction}} \cdot n_{\text{unseen}}(c_d) \cdot H_{\max}$$

其中 $n_{\text{unseen}}(c_d)$ 为预测与主导模式匹配的 unseen 物品数量。

### 统一比较

三类问题在同一信息论标尺上直接比较：

$$\text{EER}^* = \max\left(\max_{o} \text{EER}_{\text{action}}(o),\ \max_{c} \text{EER}_{\text{eliciting}}(G_c),\ \text{EER}_{\text{induction}}(G_{c_d})\right)$$

EER 框架的核心特性：
1. **无任意阈值**：三种模式的选择由信息增益的连续值比较决定，无需手动设定切换阈值
2. **自适应切换**：对话早期物品熵高、分组大，EER_eliciting 自然占优；后期分组收缩，EER_action 反超，实现从偏好探索到直接求解的平滑过渡
3. **目标定向**：EER 同时确定了 pattern 和具体的 target 物品/分组，消除了目标选择的模糊性

---

## 3.8 策略模式作为约束

**策略模式**（policy mode）不再通过 if/else 规则控制提问流程，而是被重新表述为对提问空间 $\mathcal{Q}$ 的**有效集合约束**：

$$\mathcal{Q}_{\text{valid}}(\mathcal{S}_t \mid \text{mode}) \subseteq \mathcal{Q}$$

四种策略模式的约束定义如下：

**Direct Querying（直接查询）**：只允许 action 模式，不探索用户偏好：

$$\mathcal{Q}_{\text{valid}}^{\text{direct}} = \{(\text{action}, o) : o \in \mathcal{U}_t\}$$

**User-Preference First（偏好优先）**：在偏好尚未被充分覆盖时优先使用 eliciting，全部覆盖后退化为 action：

$$\mathcal{Q}_{\text{valid}}^{\text{upf}} = \begin{cases} \{(\text{eliciting}, G_c) : |G_c| \geq 2\} & \text{if } \mathcal{P}^+ \text{ coverage} < \mathcal{U}_t \\ \{(\text{action}, o) : o \in \mathcal{U}_t\} & \text{otherwise} \end{cases}$$

**Parallel Exploration（并行探索）**：开放全部提问空间，由 EER 自主决定最优 (pattern, target)：

$$\mathcal{Q}_{\text{valid}}^{\text{parallel}} = \{(\text{action}, o)\} \cup \{(\text{eliciting}, G_c)\} \cup \{(\text{induction}, G_{c_d})\}$$

**Hybrid All（混合策略）**：同样开放全部提问空间，但引入 LLM 对 EER 进行上下文动态加权（见 3.9 节）：

$$\mathcal{Q}_{\text{valid}}^{\text{hybrid}} = \mathcal{Q}_{\text{valid}}^{\text{parallel}}$$

**联合选择**：在有效集合内找到 EER 最大的 (pattern, target) 对：

$$(\text{pattern}^*, \text{target}^*) = \arg\max_{q \in \mathcal{Q}_{\text{valid}}(\mathcal{S}_t \mid \text{mode})} \text{EER}(q \mid \mathcal{S}_t)$$

---

## 3.9 定向问题生成

确定 $(\text{pattern}^*, \text{target}^*)$ 后，提问生成器（Proposer）的职责被精确限定为：**用自然语言实现这一特定的 (pattern, target) 组合**。

不同于现有方法中 Proposer 在给定 pattern 后自由发挥选择询问对象，本框架的 Proposer 接收包含 EER 分析结果的定向指令：

```
模式：preference_eliciting
目标分组：{cotton lap blanket, soft stuffed bear}（共享预测位置 storage ottoman）
分组 EER：4.71 bits（当前最高价值目标）
unseen 泛化潜力：α=0.5，n_unseen=3 个同类物品
```

这一设计将 LLM 的作用限制在两个明确的子任务上：
1. **置信度估计**（BeliefEstimator）：对物品的放置概率分布做知识推断
2. **语言实现**（Proposer）：将确定的 (pattern, target) 转化为自然、流畅的对话语句

两个决策层（pattern 选择与 target 选择）均由确定性的 EER 计算完成，消除了 LLM 在策略决策中的随机性。

---

## 3.10 Hybrid All：LLM 加权的 EER

Parallel Exploration 与 Hybrid All 的关键区别在于：

- **Parallel Exploration**：所有提问模式的 EER 使用统一权重（$w_{\text{action}} = w_{\text{elic}} = w_{\text{ind}} = 1.0$）
- **Hybrid All**：LLM 根据对话上下文动态调整各模式的优先级权重

具体地，在每轮决策前，我们调用 LLM 估计当前对话状态下各模式的适用性：

$$\mathbf{w} = (w_{\text{action}}, w_{\text{eliciting}}, w_{\text{induction}}) = \text{LLM}_{\text{policy}}(\mathcal{S}_t)$$

最终联合得分为：

$$\text{Score}(\text{pattern}, \text{target} \mid \mathcal{S}_t) = w_{\text{pattern}} \cdot \text{EER}(\text{pattern}, \text{target} \mid \mathcal{S}_t)$$

$$(\text{pattern}^*, \text{target}^*) = \arg\max_{(\text{pattern}, \text{target}) \in \mathcal{Q}_{\text{valid}}} \text{Score}(\text{pattern}, \text{target})$$

LLM 在此承担的职责是**对话动态感知**（如用户表现出不耐烦时降低 $w_{\text{eliciting}}$，对话刚开始时提升 $w_{\text{eliciting}}$ 以快速建立偏好模型），而非直接做出模式选择决策。这使得 Hybrid All 保留了 EER 的信息论基础，同时增加了对话层面的适应性。

---

## 3.11 系统流程

完整的单轮决策流程如下：

```
S_t
 │
 ├─[BeliefEstimator]──► {P̂(c|o, S_t)} for all o ∈ U_t (∪ O_u if needed)
 │                                          │
 │                            [Shannon Entropy + Grouping]
 │                                          │
 │                             H(o), G_c^elic, G_p^ind
 │                                          │
 ├─[Policy Constraint]──► Q_valid(S_t | mode)
 │                                          │
 │                              [Joint EER Scoring]
 │                                          │
 │               [hybrid_all: LLM_policy(S_t) → w → Score = w × EER]
 │                                          │
 │                        (pattern*, target*) = argmax Score
 │                                          │
 ├─[Targeted Proposer]──► q_{t+1} (针对 target* 的 pattern* 问题)
 │
 ├─[Oracle]──────────────► a_{t+1}
 │
 └─[State Update]─────────► S_{t+1}
```

该流程在预算 $B$ 耗尽或 $\mathcal{U}_t = \emptyset$ 时终止。

---

## 3.12 框架的理论性质

**定理（EER 的自适应性）**：在物品熵均匀分布的早期阶段（$H(o) \approx H_{\max}$），若分组规模 $|G_c^{\text{elic}}| \geq 2$，则 $\text{EER}_{\text{eliciting}} \geq \text{EER}_{\text{action}}$，偏好引导模式自然占优。在物品熵集中于少数高不确定性物品的后期阶段，当 $\max_o H(o) > \sum_{o \in G_c} H(o) / |G_c^{\text{elic}}|$ 时，action 模式反超。

**推论**：在 Parallel Exploration 和 Hybrid All 模式下，本框架无需人工设定阈值即可实现从"偏好探索"到"精准确认"的自动过渡，该过渡点由对话历史和物品不确定性结构共同决定。

**Policy 可区分性**：四种 Policy Mode 在本框架下具有严格的包含关系：

$$\mathcal{Q}_{\text{valid}}^{\text{direct}} \subsetneq \mathcal{Q}_{\text{valid}}^{\text{upf}} \subsetneq \mathcal{Q}_{\text{valid}}^{\text{parallel}} = \mathcal{Q}_{\text{valid}}^{\text{hybrid}}$$

Parallel 与 Hybrid 的区分不在于有效集合的大小，而在于 EER 权重的来源（统一 vs LLM 动态）。这一差异在信息论层面是清晰的：Parallel 是无先验的信息最大化，Hybrid 是以 LLM 对话策略为先验的信息最大化。

---

## 3.13 与现有策略框架的关系

本框架与论文中四种 Policy Mode 的关系如下：

| Policy Mode | 核心目标 | 框架表达 |
|---|---|---|
| Direct Querying | 最短路径解决当前最不确定物品 | 约束至 action 子空间，取 argmax H(o) |
| User-Preference First | 优先建立用户偏好模型再求解 | 分阶段约束：先 eliciting 子空间，后 action |
| Parallel Exploration | 动态平衡探索（偏好）与利用（直接问） | 全空间 EER 自主选择，无需人工阈值 |
| Hybrid All | 利用 LLM 的对话策略知识调节探索深度 | 全空间 + LLM 权重调节，EER 保底 |

**Pattern 选择方式对比**（三种实现路径）：

| 实现方式 | 决策机制 | 优点 | 缺点 |
|---|---|---|---|
| 规则（Rule） | 硬编码的阈值和 if/else | 低延迟，可解释，稳定 | 需人工调参，无法适应数据分布变化 |
| 熵（EER） | 本框架，联合 (pattern, target) | 无阈值，自适应，信息论基础，可解释 | 需要 BeliefEstimator 调用 |
| LLM 直接推断 | LLM 根据对话历史选 pattern | 最灵活，具备对话感知 | 高方差，不可解释，延迟高 |

---

## 附录：代码映射（实现前预规划）

| 论文概念 | 目标代码文件 | 目标类/函数 |
|---|---|---|
| $\mathcal{Q}_{\text{valid}}(\mathcal{S}_t \mid \text{mode})$ | `question_policy.py` | `PolicyMode.valid_question_space()` |
| Joint EER 计算 | `question_policy.py` | `_joint_eer_select()` |
| EER_action | `question_policy.py` | `_eer_action()` |
| EER_eliciting | `question_policy.py` | `_eer_eliciting()` |
| EER_induction | `question_policy.py` | `_eer_induction()` |
| LLM 权重（Hybrid All）| `question_policy.py` | `_llm_weight()` |
| 定向 Proposer | `proposers.py` | `{Action,Eliciting,Induction}Proposer.generate(target=...)` |
| 置信度估计 | `belief_estimator.py` | `BeliefEstimator.estimate_detailed()` |
| unseen 估计 | `belief_estimator.py` | `BeliefEstimator.estimate_detailed(include_unseen=True)` |

---

## 待解决的开放问题

1. **eliciting 分组的粒度**：当前用 Top-1 预测位置分组，但"位置"不等于"类别"。是否应使用 LLM 对物品做显式类别归属判断？（代价：额外一次 LLM 调用）

2. **induction 的多个候选模式**：当前只考虑 $c_d$（最频繁的一个主导位置），但 $\mathcal{A}^+$ 中可能同时存在多个有效模式（如书架和衣柜）。是否应对所有候选模式分别计算 EER 再取最大？

3. **EER 的近似假设**：我们假设"一次回答可完全解决对应分组的不确定性"，实际上用户可能拒绝、部分确认或给出模糊回答。是否需要引入"回答成功概率" $p_{\text{success}}$ 折扣系数？

4. **Hybrid All 的 LLM 权重实现**：LLM 输出 $(w_{\text{action}}, w_{\text{elic}}, w_{\text{ind}})$ 的 prompt 设计尚未确定。权重是 [0,1] 范围内的独立值还是 Softmax 归一化的分布？

5. **完全理论驱动 vs 混合**：若 EER 联合框架在 Parallel Exploration 中已经自动实现最优切换，那么 User-Preference First 的约束是否仍然必要，还是可以由 EER 自然复现其行为？（实验上可验证）
