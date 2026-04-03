# 基于信息熵的不确定性驱动提问策略

## 3.1 问题形式化

给定一个家居整理场景，智能体需要在有限的提问预算 $B$ 内，通过与用户的多轮对话，推断出所有物品的正确摆放位置。形式化地，场景由元组 $(R, \mathcal{C}, \mathcal{O}_s, \mathcal{O}_u)$ 定义，其中 $R$ 为房间名，$\mathcal{C} = \{c_1, \dots, c_M\}$ 为可用收纳位置集合，$\mathcal{O}_s$ 和 $\mathcal{O}_u$ 分别为用户可见物品（seen objects）与不可见物品（unseen objects）。智能体在每一轮对话中选择一种提问模式（question pattern）并生成具体问题，根据用户回答更新内部状态，最终为所有物品给出摆放预测。

## 3.2 智能体状态表示

我们维护一个结构化的智能体状态 $\mathcal{S}_t$，在第 $t$ 轮对话后包含以下要素：

- **已确认放置** $\mathcal{A}^+ = \{(o_i, c_j)\}$：经用户确认的物品-位置对
- **已确认偏好** $\mathcal{P}^+ = \{(h_k, \mathcal{O}_k, c_k)\}$：已确认的用户偏好规则，其中 $h_k$ 为偏好假设的自然语言描述，$\mathcal{O}_k$ 为该偏好覆盖的物品集合，$c_k$ 为对应的收纳位置
- **否定证据** $\mathcal{A}^-$, $\mathcal{P}^-$：被用户否定的放置与偏好
- **未解决物品** $\mathcal{U}_t \subseteq \mathcal{O}_s$：尚未获得确定摆放位置的可见物品
- **问答历史** $\mathcal{Q}_t = \{(q_i, a_i)\}_{i=1}^{t}$：已进行的对话记录

## 3.3 提问模式

我们定义三种互补的提问模式（question patterns），每种模式获取不同粒度的用户偏好信息：

**Action-Oriented Questioning（直接提问）** 针对单个未解决物品 $o \in \mathcal{U}_t$ 直接询问其摆放位置，获取确定性的物品-位置映射。该模式信息增益集中但覆盖范围有限，单次提问至多解决一个物品的不确定性。

**Preference Eliciting（偏好引导）** 询问用户关于某一类别物品的高层偏好规则（如"电子设备通常放在哪里？"）。用户的回答可同时覆盖多个物品的放置决策，具有较高的信息效率，但依赖于对类别结构的准确假设。

**Preference Induction（偏好归纳）** 基于已收集的行为证据 $\mathcal{A}^+$，向用户提出归纳性假设（如"我注意到您把书籍都放在了书架上，这是一个通用规则吗？"），以确认或修正从观察中提炼的偏好模式。该模式要求已有足够的行为证据作为归纳基础。

## 3.4 基于 LLM 的置信度估计

不确定性驱动策略的核心在于对当前状态下每个未解决物品的摆放不确定性进行量化。我们利用大语言模型作为置信度估计器（Belief Estimator），在每一轮决策前对所有未解决物品的摆放可能性进行评估。

具体地，给定当前智能体状态 $\mathcal{S}_t$，我们通过结构化输出提示 LLM 生成每个未解决物品 $o \in \mathcal{U}_t$ 的放置信念分布：

$$\hat{P}(c \mid o, \mathcal{S}_t) \quad \forall c \in \mathcal{C}$$

为降低结构化输出的难度并提高稳定性，我们不要求模型输出完整的 $M$ 维概率分布，而仅要求输出 Top-3 最可能的收纳位置及其概率估计 $\{(c_1, p_1), (c_2, p_2), (c_3, p_3)\}$，其中 $\sum_{k=1}^{3} p_k \leq 1$。剩余概率质量 $1 - \sum p_k$ 均匀分配至其他 $M-3$ 个收纳位置。

为改善 LLM 的概率校准质量，我们在提示中引入显式的校准指引：对于常识性明显的放置（如"书籍→书架"），要求 Top-1 概率在 0.70–0.90 范围；对于证据不足的物品，要求概率均匀分散（Top-1 在 0.25–0.35 范围）。实验表明，该策略有效缓解了 LLM 输出概率趋于模板化的问题。

## 3.5 Shannon 熵计算与目标选择

基于估计的置信度分布，我们计算每个未解决物品的 Shannon 熵作为其放置不确定性的量化指标：

$$H(o) = -\sum_{c \in \mathcal{C}} \hat{P}(c \mid o, \mathcal{S}_t) \log_2 \hat{P}(c \mid o, \mathcal{S}_t)$$

熵值越高表明智能体对该物品的放置越不确定，因此该物品越值得被提问。最大熵为 $H_{\max} = \log_2 M$（对应完全均匀分布），最小熵为 0（对应完全确定放置）。

## 3.6 熵驱动的模式选择

我们不使用预设阈值或硬编码规则来选择提问模式，而是引入期望熵减（Expected Entropy Reduction, EER）作为统一度量，让三种提问模式在同一信息论标尺上直接竞争。

利用 3.4 节中 LLM 输出的置信度估计，我们同时获得每个物品的 Shannon 熵 $H(o)$ 以及其 Top-1 预测位置 $\hat{c}(o) = \arg\max_c \hat{P}(c \mid o, \mathcal{S}_t)$。基于此，我们分别计算三种模式的期望熵减：

**Action-Oriented 的 EER。** 直接提问可解决单个物品的不确定性，提问后该物品的熵降至零。因此，最优行动的期望熵减等于最高熵物品的熵值：

$$\text{EER}_{\text{action}} = H(o^*), \quad o^* = \arg\max_{o \in \mathcal{U}_t} H(o)$$

**Preference Eliciting 的 EER。** 偏好引导问题作用于一组共享相同摆放偏好的物品。我们利用 Top-1 预测位置进行自然分组：共享同一个 $\hat{c}(o)$ 的物品可能被同一条偏好规则覆盖。一条偏好问题的期望熵减等于最优组的总熵：

$$\text{EER}_{\text{eliciting}} = \max_{c \in \mathcal{C}} \sum_{o \in G_c} H(o), \quad G_c = \{o \in \mathcal{U}_t : \hat{c}(o) = c,\ |G_c| \geq 2\}$$

**Preference Induction 的 EER。** 偏好归纳利用已确认的行为证据来建立一般性规则。我们从未归纳的 confirmed actions 中找到主导收纳位置 $c_d$（出现频率最高的位置），并计算与之匹配的未解决物品的总熵：

$$\text{EER}_{\text{induction}} = \sum_{o \in \mathcal{U}_t : \hat{c}(o) = c_d} H(o)$$

其中 $c_d = \arg\max_c |\{a \in \mathcal{A}^+_{\text{unsummarized}} : a.\text{receptacle} = c\}|$。

**模式选择规则。** 在当前策略允许的模式集合中，选择 EER 最高的模式：

$$\text{pattern}^* = \arg\max_{\text{pattern} \in \text{allowed}} \text{EER}_{\text{pattern}}$$

该设计具有以下特性：（1）**无任意阈值**——三种模式的选择完全由信息增益的连续值比较决定；（2）**自适应切换**——对话早期多个物品不确定性高、分组大，EER_eliciting 自然占优；后期不确定物品减少、分组收缩，EER_action 反超，实现从偏好探索到直接求解的平滑过渡；（3）**模式无关**——EER 框架适用于所有策略配置，通过 allowed patterns 约束即可即插即用。

## 3.7 系统流程

完整的单轮决策流程如下：

$$\mathcal{S}_t \xrightarrow{\text{Belief Estimator}} \{\hat{P}(\cdot \mid o, \mathcal{S}_t)\}_{o \in \mathcal{U}_t} \xrightarrow{\text{EER Selector}} (\text{pattern}^*, \text{guidance}) \xrightarrow{\text{Proposer}} q_{t+1} \xrightarrow{\text{Oracle}} a_{t+1} \xrightarrow{\text{State Update}} \mathcal{S}_{t+1}$$

其中 guidance 是传递给下游 Proposer 的自然语言指令，包含 EER 分析结果（如"Objects like cotton lap blanket, soft stuffed bear share uncertainty toward 'storage ottoman' (group EER=4.71 bits)"），引导 Proposer 生成针对最有信息价值的目标的提问。该流程在预算 $B$ 耗尽或所有物品均已解决时终止。

---

## 代码映射

| 论文概念 | 代码文件 | 核心类/函数 |
|---------|---------|------------|
| 智能体状态 $\mathcal{S}_t$ | `agent_schema.py` | `AgentState` TypedDict |
| 置信度估计 $\hat{P}(c \mid o, \mathcal{S}_t)$ | `belief_estimator.py` | `BeliefEstimator.estimate()` |
| Shannon 熵 $H(o)$ | `belief_estimator.py` | `shannon_entropy()` |
| EER 模式选择 | `question_policy.py` | `QuestionPolicyController._entropy_select()` |
| 三种提问模式 Proposer | `proposers.py` | `ActionProposer`, `PreferenceElicitingProposer`, `PreferenceInductionProposer` |
| 状态更新 $\mathcal{S}_t \to \mathcal{S}_{t+1}$ | `state_update.py` | `StateUpdate` |
| 最终放置预测 | `evaluation.py` | `FinalPlacementPlanner` |

## 实验验证摘要（5-episode, budget=[3,6]）

| 模式 | Budget | Rule (seen/unseen) | Entropy (seen/unseen) | $\Delta$ seen | $\Delta$ unseen |
|------|--------|-------------------|----------------------|--------------|----------------|
| direct_querying | 3 | 0.433 / 0.450 | 0.517 / 0.417 | +0.084 | -0.033 |
| direct_querying | 6 | 0.633 / 0.517 | 0.717 / 0.617 | +0.084 | +0.100 |
| user_preference_first | 3 | 0.517 / 0.467 | 0.633 / 0.483 | +0.116 | +0.016 |
| user_preference_first | 6 | 0.767 / 0.650 | 0.833 / 0.633 | +0.066 | -0.017 |
| parallel_exploration | 3 | 0.433 / 0.417 | 0.517 / 0.417 | +0.084 | +0.000 |
| parallel_exploration | 6 | 0.600 / 0.433 | 0.717 / 0.617 | +0.117 | +0.184 |
| hybrid_all | 3 | 0.533 / 0.467 | 0.633 / 0.483 | +0.100 | +0.016 |
| hybrid_all | 6 | 0.667 / 0.550 | 0.783 / 0.567 | +0.116 | +0.017 |
| **平均** | | **0.573 / 0.494** | **0.669 / 0.529** | **+0.096** | **+0.035** |

