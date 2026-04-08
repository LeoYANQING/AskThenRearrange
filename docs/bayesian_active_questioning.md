# 基于贝叶斯理论的主动提问框架：技术报告

## 1. 为什么要超越 LLM + 规则？

当前 AskThenRearrange 系统的每个决策环节（信念估计、问题选择、状态更新、放置规划）都依赖 **LLM-as-module** 的架构。问题策略通过一个启发式的 Expected Entropy Reduction (EER) 来选择三种提问模式（AO/PE/PI），核心计算是对候选对象的 Shannon 熵求和，再加手工设置的泛化权重（PE 的 α=0.5，PI 的 α=1.0）。

**现有方案的根本局限：**

| 维度 | 当前方案 | 贝叶斯替代方案 |
|------|---------|--------------|
| 信念表示 | 每轮重新调用 LLM 估计 | 显式后验 P(c\|o, D_t)，增量更新 |
| 问题选择 | EER = 熵之和 + 启发式奖励 | 期望信息增益 EIG = H[后验] − E_answer[H[后验\|answer]] |
| 答案建模 | 不预测 P(answer\|question) | 似然模型支持前瞻推理 |
| 策略选择 | 基于规则的模式切换 | 在不同查询类型间直接比较 EIG |
| 泛化到 unseen | 推理阶段的 LLM 语义类比 | 类别成员概率 P(o∈k) 自动传播更新 |
| 停止准则 | 固定预算 B | 信息价值 VOI：当 EIG < 代价时停止 |

**核心洞察：每种提问类型（AO/PE/PI）有不同的信息结构——它们以不同方式更新后验的不同部分。** 贝叶斯框架使这一点显式化，从而实现跨类型的原则性比较。

---

## 2. 问题形式化

### 2.1 放置分布

设：
- O = {o₁, ..., o_N}：所有物体（seen + unseen）
- C = {c₁, ..., c_M}：所有容器（receptacles）
- D_t = {(q₁,a₁), ..., (q_t,a_t)}：t 轮对话后的历史记录
- π: O → C：真实（未知）放置函数

我们需要维护放置后验信念：

**P(π(oᵢ) = cⱼ | D_t)　　∀ oᵢ ∈ O, cⱼ ∈ C**

这是一个 N×M 的分配矩阵，每行是一个关于 M 个容器的 categorical 分布。

### 2.2 先验：P(π | context)

在任何提问之前，我们有一个**上下文先验**：

**P₀(cⱼ | oᵢ) = P(cⱼ | oᵢ, R, C)**

其中 R 是房间类型。先验编码了：
- **语义兼容性**："炒锅大概率放厨房橱柜"
- **房间规范**："卧室有床头柜，没有灶台"
- **物体-容器共现统计**：来自常识或学习到的嵌入

**实现方式 1（LLM）**：在每个 episode 开始时调用一次 `belief_estimator` 生成 P₀。

**实现方式 2（Embedding）**：用句向量编码物体和容器，通过 softmax 相似度：

```
P₀(cⱼ | oᵢ) = exp(sim(e_oᵢ, e_cⱼ) / τ) / Σ_k exp(sim(e_oᵢ, e_cₖ) / τ)
```

这种方式更快、可微分、不需要每轮都调用 LLM。

### 2.3 类别结构——连接物体层与规则层的关键

当前系统缺少一个显式的**类别模型**。定义：

- K = {k₁, ..., k_L}：潜在物体类别（如 "电子配件"、"阅读材料"、"日常厨具"）
- P(oᵢ ∈ k_l)：类别成员概率
- P(cⱼ | k_l)：类别级放置分布

联合模型变为一个**混合模型**：

**P(π(oᵢ) = cⱼ) = Σ_l P(oᵢ ∈ k_l) · P(cⱼ | k_l)**

**为什么这对多粒度提问至关重要：**
- **AO** 直接更新 P(cⱼ | oᵢ)（物体级）
- **PE** 更新 P(cⱼ | k_l)（类别级），传播到类别 k_l 中的**所有**物体——包括 unseen
- **PI** 确认或否定某个特定的 P(cⱼ | k_l) 假设，将软信念转为硬分配

以数据集中的一个实例说明：

```json
// Episode 0: living room, 7 receptacles
// annotator_notes:
//   "Handheld media-control and portable entertainment devices go to the media drawer."
//   "Upright reading and reference books go to the bookshelf."
//   ...
// seen_objects: ["battery-powered remote", "wireless speaker", "hardcover novel", ...]
// unseen_objects: ["bluetooth headphones", "paperback mystery", ...]
```

在这个场景中：
- 类别 k₁ = "手持电子设备" → 成员：remote, speaker, (unseen) headphones
- 类别 k₂ = "阅读材料" → 成员：novel, art book, (unseen) paperback

一个 PE 问题 "你通常怎么整理手持电子设备？" 得到 "放 media drawer"，就能同时更新 remote、speaker 和 unseen 的 headphones 的后验——这就是 PE 能在 unseen 上大幅超越 AO 的数学原因。

---

## 3. 三种提问类型的贝叶斯更新机制

### 3.1 Action-Oriented (AO) 更新

**问题**："wireless speaker 放哪里？"  
**回答**："放 media drawer"

**似然函数**：
```
P(a=cⱼ | π(oᵢ)=cₖ) = { 1−ε  if k=j;  ε/(M−1)  otherwise }
```

ε 建模回答噪声（用户说 cⱼ 但真实偏好可能因歧义而是 cₖ）。

**后验更新**（单物体，Bayes 规则）：
```
P(π(oᵢ)=cₖ | a=cⱼ) ∝ P(a=cⱼ | π(oᵢ)=cₖ) · P_t(π(oᵢ)=cₖ)
```

**信息结构**：AO 只更新 N×M 矩阵的**一行**，其余不变。

**期望信息增益**：
```
EIG_AO(oᵢ) = H[π(oᵢ)] − Σ_cⱼ P(π(oᵢ)=cⱼ) · H[π(oᵢ) | a=cⱼ]
```

由于 AO 回答几乎是确定性的（指向一个容器），EIG_AO ≈ H[π(oᵢ)]，即物体 oᵢ 当前的熵。这与当前的启发式一致：选熵最高的物体。但关键区别是 EIG_AO 只捕获对 oᵢ 的信息增益——**不会泛化到其他物体**。

### 3.2 Preference Eliciting (PE) 更新

**问题**："你通常怎么整理电子配件？"  
**回答**："放 media drawer"

**似然函数**（类别级）：
```
P(a=cⱼ | ∀o∈k_l: π(o)=cₖ) = { 1−ε  if k=j;  ε/(M−1)  otherwise }
```

**后验更新**（类别级，传播到所有成员）：
```
P(π(oᵢ)=cₖ | a=cⱼ) = Σ_l P(oᵢ∈k_l) · P(cₖ|k_l, a) + P(oᵢ∉任何k) · P_t(π(oᵢ)=cₖ)
```

**信息结构**：PE 更新整个**类别列**。对 P(oᵢ∈k_l) 高的物体，更新强烈；低的物体，更新微弱。这自然处理了：
- 明确属于该类别的 seen 物体 → 强更新
- 可能属于该类别的 unseen 物体 → 中等更新
- 与该类别无关的物体 → 不更新

**期望信息增益**（PE 优势的数学来源）：
```
EIG_PE(k_l) = Σ_{oᵢ∈O} P(oᵢ∈k_l) · EIG_{oᵢ从类别回答}
```

求和遍历**所有物体**（seen + unseen），以成员概率加权。当类别有 2+ 成员时，这严格大于任何单个 AO 的 EIG——这从数学上证明了 PE 在 unseen 上超越 AO 的原因。

### 3.3 Preference Induction (PI) 更新

**问题**："我注意到你把书和杂志放在书架上——这是一个通用规则吗？"  
**回答**："是" / "不是" / "是的，但...有例外"

**假设的先验**：基于已确认的 actions A⁺ 中指向 c₁ 的比例：
```
P(k_l→c₁ | A⁺) = count(c₁ in A⁺_{k_l}) / count(A⁺_{k_l})
```

**信息结构**：PI 是一个**二值验证**问题，EIG 由以下公式约束：
```
EIG_PI(k_l, c) = H_binary(P(k_l→c)) · |{o : P(o∈k_l) > θ}|
```

其中 H_binary 是二值变量的熵（最大 1 bit）。PI 在以下条件下最有价值：
1. 假设具有中等置信度（既不太确定也不太不确定）
2. 类别覆盖多个物体
3. 类别包含 unseen 物体

---

## 4. 统一问题选择：跨类型 EIG 比较

贝叶斯方法的核心算法贡献是**直接在不同提问类型间比较 EIG**：

```
q* = argmax_{q ∈ Q_AO ∪ Q_PE ∪ Q_PI} EIG(q)
```

其中：
- Q_AO = {询问物体 oᵢ : oᵢ ∈ unresolved}
- Q_PE = {询问类别 k_l : k_l 有 ≥2 成员}
- Q_PI = {确认假设 (k_l, cⱼ) : 有证据支持}

### 4.1 EIG 随预算的演变图景

| 预算阶段 | EIG_AO | EIG_PE | EIG_PI | 最优策略 |
|---------|--------|--------|--------|---------|
| B=0（开始） | 高（全部不确定） | **极高**（大量未覆盖类别） | 0（无 PI 证据） | PE |
| B=1-2 | 中（部分已解决） | 高（仍有未覆盖类别） | 低（少量确认动作） | PE 或 AO |
| B=3-4 | 中低 | 中（类别缩小） | 上升（证据积累） | PE/PI/AO |
| B=5-6 | 低（少数未解决） | 低（多数类别已覆盖） | 中（模式清晰） | AO 或 PI |

这解释了为什么 **UPF（先偏好后动作）策略**——先用 PE 再切 AO——是 EIG 最优策略的自然近似。也解释了为什么能调用 PI 的混合策略应该表现最好。

---

## 5. 每种策略模式下的先验与后验知识

### 5.1 DQ (Direct Querying) —— 纯 AO

**先验**：P₀(cⱼ|oᵢ) 来自上下文（房间类型 + 物体名 + 容器名）

**DQ 学到了什么**：被询问物体的个体放置概率 P(π(oᵢ)=cⱼ|D_t)

**DQ 没学到什么**：类别结构。没有 P(cⱼ|k_l) 被估计。Unseen 物体的放置完全依赖先验 P₀ 加上最终 planner 能提取的语义类比。

**B 轮提问后的后验**：
- 被询问的物体：近 delta 分布（高确定性）
- 未被询问的 seen 物体：与先验相同（无信息传递）
- Unseen 物体：与先验相同

**数学刻画**：
```
H_total^DQ(B) = Σ_{queried} H_ε(≈0) + Σ_{unqueried} H[π(oᵢ)](不变) + Σ_{unseen} H[π(oᵢ)](不变)
```

总剩余熵恰好减少 B × H_avg(被询问物体)，**无溢出效应**。

### 5.2 PE (Preference Eliciting)

**先验**：P₀(cⱼ|oᵢ) + 隐式类别结构 P(oᵢ∈k_l)

**PE 学到了什么**：类别级规则 P(cⱼ|k_l)，传播到所有类别成员

**PE 没学到什么**：个体例外（偏离类别规则的物体）

**B 轮提问后的后验**（每个问题覆盖平均大小 s̄ 的类别）：
```
H_total^PE(B) ≈ H_total^prior − B · s̄ · H_avg
```

由于 s̄ > 1 且求和包含 unseen 物体，PE 每个问题移除的总熵多于 DQ。

### 5.3 UPF (User Preference First) —— PE→AO

**阶段 1（PE）**：学习类别级规则。t_switch 轮 PE 之后：
- 大多数大类别已被覆盖
- 剩余物体要么是类别边界案例，要么是独特物品

**阶段 2（AO）**：逐个解决残余高熵物体。

**贝叶斯 UPF 相对于规则 UPF 的优势**：不需要硬编码切换点。EIG 计算**自动识别**何时从类别级切换到物体级提问：

**切换条件**：当 max_k EIG_PE(k) < max_o EIG_AO(o) 时，自然过渡。

### 5.4 Hybrid（全 EIG 优化）

```
q* = argmax over {EIG_PE(k_l)} ∪ {EIG_AO(oᵢ)} ∪ {EIG_PI(k_l, cⱼ)}
```

同时考虑所有提问类型。PI 在 AO 积累足够确认动作后变得可用。EIG 决定确认假设（PI）是否比提出新类别问题（PE）或解决具体物体（AO）更有价值。

---

## 6. 端到端贝叶斯架构

### 6.1 流程总览

```
Episode 开始
    │
    ▼
[先验初始化]  P₀(cⱼ|oᵢ) ∀物体
              P(oᵢ∈k_l) 类别成员概率
    │
    ▼
[提问循环]  for t = 1, ..., B:
    │
    ├─ [EIG 计算]  计算所有候选问题的 EIG
    │     ├─ EIG_AO(oᵢ) ∀未解决物体
    │     ├─ EIG_PE(k_l) ∀未覆盖类别
    │     └─ EIG_PI(k_l, c) ∀有证据的假设
    │
    ├─ [问题选择]  q* = argmax EIG
    │
    ├─ [问题生成]  LLM 将 q* 转为自然语言问句
    │
    ├─ [用户回答]  (oracle 或真实用户)
    │
    └─ [贝叶斯更新]  更新 P(cⱼ|oᵢ) ∀受影响物体
          ├─ AO: 更新单物体行
          ├─ PE: 更新类别列 → 传播到所有成员
          └─ PI: 类别列的二值更新
    │
    ▼
[放置决策]  ∀物体，分配 argmax_c P(c|oᵢ, D_B)
```

### 6.2 核心模块设计

#### 模块 1：先验估计器

```python
class BayesianPrior:
    def __init__(self, room, objects, receptacles):
        # 方式 1：embedding 相似度
        self.placement_prior = softmax_similarity(objects, receptacles, tau=0.5)  # N × M
        
        # 类别发现：LLM 一次性调用
        self.categories, self.membership = discover_categories(objects)  # L 类别, N × L
        
        # 类别级先验：聚合物体级先验
        self.category_prior = aggregate(self.placement_prior, self.membership)  # L × M
```

#### 模块 2：EIG 计算器

```python
class EIGCalculator:
    def eig_ao(self, object_idx):
        """AO 的 EIG ≈ 物体当前熵（闭式解）"""
        return entropy(self.posterior[object_idx])
    
    def eig_pe(self, category_idx):
        """PE 的 EIG = 在所有可能回答上的边际化"""
        eig = 0
        for c_j in range(M):
            p_answer = self.category_prior[category_idx, c_j]
            post_after = self.simulate_pe_update(category_idx, c_j)
            kl_sum = sum(
                self.membership[o, category_idx] *
                kl_divergence(post_after[o], self.posterior[o])
                for o in range(N)
            )
            eig += p_answer * kl_sum
        return eig
    
    def eig_pi(self, category_idx, receptacle_idx):
        """PI 的 EIG = 二值熵 × 受影响物体数 × 平均熵减"""
        p_yes = self.category_prior[category_idx, receptacle_idx]
        h_binary = -p_yes * log2(p_yes) - (1-p_yes) * log2(1-p_yes)
        affected = [o for o in range(N) if self.membership[o, category_idx] > threshold]
        avg_reduction = mean(entropy(self.posterior[o]) for o in affected)
        return h_binary * len(affected) * avg_reduction
```

#### 模块 3：贝叶斯状态更新器

```python
class BayesianUpdater:
    def update_ao(self, object_idx, answered_receptacle):
        """直接 Bayes 更新：单行"""
        likelihood = np.full(M, epsilon / (M-1))
        likelihood[answered_receptacle] = 1 - epsilon
        self.posterior[object_idx] *= likelihood
        self.posterior[object_idx] /= self.posterior[object_idx].sum()
    
    def update_pe(self, category_idx, answered_receptacle):
        """类别级更新：传播到所有成员"""
        self.category_prior[category_idx] = near_delta(answered_receptacle, epsilon)
        for o in range(N):
            w = self.membership[o, category_idx]
            self.posterior[o] = (1-w) * self.posterior[o] + w * self.category_prior[category_idx]
            self.posterior[o] /= self.posterior[o].sum()
    
    def update_pi(self, category_idx, receptacle_idx, confirmed: bool):
        """PI 更新：确认则等同 PE，否定则压制假设"""
        if confirmed:
            self.update_pe(category_idx, receptacle_idx)
        else:
            self.category_prior[category_idx, receptacle_idx] *= delta  # delta << 1
            self.category_prior[category_idx] /= self.category_prior[category_idx].sum()
            for o in range(N):
                w = self.membership[o, category_idx]
                self.posterior[o] = (1-w) * self.posterior[o] + w * self.category_prior[category_idx]
                self.posterior[o] /= self.posterior[o].sum()
```

---

## 7. 类别发现与成员估计

这是整个框架最关键的组件——类别质量直接决定 PE/PI 的效果。

### 7.1 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| LLM 直接生成类别 + 成员 | 语义理解强 | 不稳定，每次调用结果可能不同 |
| Embedding 聚类 (GMM/soft k-means) | 确定性、覆盖 unseen | 可能产生语义不连贯的类别 |
| **混合方案**：LLM 生成类别标签，Embedding 计算成员概率 | 两者优点兼得 | 需要 embedding 模型 |

### 7.2 推荐：混合方案

```
步骤 1：LLM 发现类别标签
  Input: room="living room", objects=["battery-powered remote", "wireless speaker", ...]
  Output: categories=["portable electronics", "reading materials", "soft furnishings", ...]

步骤 2：Embedding 计算成员概率
  P(oᵢ ∈ k_l) = softmax(sim(embed(oᵢ), embed(k_l)) / τ)
  
  示例：
    P("wireless speaker" ∈ "portable electronics") = 0.82
    P("wireless speaker" ∈ "reading materials") = 0.01
    P("bluetooth headphones" ∈ "portable electronics") = 0.78  ← unseen 也被覆盖
```

**关键优势**：unseen 物体自然获得类别成员概率，无需额外处理。

---

## 8. 与现有数据集的对接

以数据集第一个 episode 为例：

```
Room: living room
Receptacles: [bookshelf, coffee table, side table, media drawer, TV stand, 
              storage ottoman, storage basket]  (M=7)
Seen objects: 12个    Unseen objects: 12个    N=24
Annotator notes: 6条规则，自然对应 ≈6 个类别
```

**annotator_notes 自然映射为类别先验**：

| Annotator Note | 隐含类别 k_l | 隐含容器 c | Seen 成员 | Unseen 成员 |
|---------------|-------------|-----------|----------|------------|
| "Handheld media-control..." | 手持电子设备 | media drawer | remote, speaker | (headphones等) |
| "Large wired entertainment..." | 大型有线设备 | TV stand | game console, reading light | (DVD player等) |
| "Upright reading and reference..." | 阅读材料 | bookshelf | novel, art book | (paperback等) |
| "Soft comfort items..." | 柔软舒适物品 | storage ottoman | lap blanket, stuffed bear | (throw pillow等) |
| "Drinkware, small tabletop..." | 桌面物品 | side table | ceramic mug, water glass | (coaster等) |
| "Stored, seasonal..." | 季节/存储物品 | storage basket | garland, coaster pouch | (ornament等) |

**从贝叶斯视角看当前数据集**：
- 每个 episode 有 5-8 个容器，对应约 5-8 个潜在类别
- 每个类别平均覆盖 2-4 个 seen 物体 + 2-4 个 unseen 物体
- 一个 PE 问题能覆盖 ~3 个物体（seen + unseen），所以 B=6 的 PE 可以覆盖 ~18/24 = 75% 的物体

这与实验观察一致：PE 在 B=6 时达到 82.3% seen / 71.9% unseen 准确率。

---

## 9. 与核心文献的联系

### 9.1 BALD (Houlsby et al., 2011) 与我们的 EIG

我们的 EIG_AO 等价于 BALD 的采集函数应用于 categorical 分类。BALD 选择后验预测最不确定的点——对我们来说就是放置最不确定的物体。**我们的扩展在于 PE**：BALD 不考虑同时更新多个数据点的查询。

### 9.2 OPEN (Handa et al., 2024)

OPEN 是与我们系统最接近的架构匹配：贝叶斯实验设计层做查询选择 + LLM 层做自然语言生成。关键差异：
- OPEN 使用粒子滤波近似；我们可以用精确 categorical 后验（有限容器集）
- OPEN 聚焦成对偏好；我们有三种查询类型，信息结构各不相同
- OPEN 的 LM 生成特征；我们的 LLM 生成实际问题

### 9.3 APRICOT (Wang et al., 2024, CoRL)

APRICOT 做机器人整理场景中的主动偏好学习（高度相关）。关键差异：
- APRICOT 从演示+主动查询中学习；我们纯粹从对话中学习
- APRICOT 不形式化多粒度查询——这是我们的核心贡献点

### 9.4 Query Type Matters (Huang et al., 2015, IJCAI)

核心洞察：选择正确的查询**类型**比选择正确的实例更重要。他们在多标签主动学习中表明，查询全部标签 vs 单个标签 vs 标签对的选择影响大于选择哪个实例。我们的 AO/PE/PI 区分是家居整理场景中的对应物。

### 9.5 Asking Easy Questions (Biyik et al., 2019, CoRL)

引入了查询信息量应与认知代价平衡的理念。在我们的框架中：

```
q* = argmax_q [ EIG(q) − λ · CognitiveCost(q) ]
```

其中 AO 认知代价低但信息少，PE 认知代价中但信息多，PI 认知代价可变。λ 是用户特定参数，可从交互模式中学习——这直接联系到 IJHCS 论文中的 RQ4（organizing expertise 的调节作用）。

### 9.6 Sadigh/Biyik 系列 (RSS 2017 → CoRL 2018/2019 → RSS 2020)

这个系列从 Bradley-Terry 偏好模型出发，逐步发展了：
- 主动偏好学习的 maximum volume removal (2017)
- 批量主动偏好查询 (2018)
- 平衡信息量和认知代价的查询设计 (2019)
- 非线性 GP 偏好模型 (2020)
- 多源反馈整合 (2022, IJRR)

我们的工作将这条线从**成对轨迹比较**扩展到**多粒度自然语言提问**。

### 9.7 POMDP 偏好引导 (Boutilier, 2002, AAAI)

我们的问题策略在形式上是一个 POMDP：
- **状态**：真实放置函数 π（隐藏）
- **信念**：后验 P(π|D_t)
- **动作**：来自 Q_AO ∪ Q_PE ∪ Q_PI 的问题
- **观测**：用户回答
- **奖励**：终止时的放置准确率

近视 EIG 策略（贪心单步前瞻）是最优 POMDP 策略的常用近似。

---

## 10. 实现路线图

### 阶段 1：后验追踪（替换 belief_estimator.py）

1. 实现 `BayesianPrior`：embedding-based P₀
2. 实现 `BayesianUpdater`：精确 categorical 更新
3. 验证：B=6 AO 问题后的后验应匹配当前 confirmed_actions 准确率

### 阶段 2：EIG 问题选择（替换 question_policy.py EER）

1. 实现 `EIGCalculator`：闭式 EIG_AO
2. 实现 EIG_PE（对可能回答的边际化）
3. 实现 EIG_PI（二值假设检验）
4. 验证：EIG 选择的问题应平均优于熵和 EER

### 阶段 3：类别模型（增强 proposers.py）

1. 实现 `CategoryDiscovery`：LLM + embedding 混合方案
2. 每 episode 缓存类别；每次回答后更新成员概率
3. 验证：类别成员概率应比语义类比更好地预测 unseen 物体放置

### 阶段 4：端到端评估

1. 在全部 102 episodes 上运行贝叶斯系统，budgets 0-6
2. 与当前 LLM+规则系统在相同 episodes 上对比
3. 消融实验：Bayesian-DQ vs Bayesian-PE vs Bayesian-UPF vs Bayesian-Hybrid
4. 分析：EIG 与准确率的相关性、类别质量、切换点分析

---

## 11. 预期优势与风险

### 优势

1. **原则性策略选择**：无手工 α 权重或模式切换规则。跨问题类型的 EIG 比较自动决定最优策略。

2. **增量更新**：无需每轮从零重估信念。后验自然积累证据。

3. **Unseen 处理**：类别成员概率 P(o∈k) 自动将 PE/PI 更新传播到 unseen 物体，无需在推理时依赖 LLM 语义类比。

4. **停止准则**：当 max EIG < 阈值，agent 知道额外提问不会显著改善放置。可实现自适应预算分配。

5. **可解释性**：后验 P(c|o, D_t) 精确展示每个放置为什么被选择，以及每个问题贡献了多少信息。

### 风险

1. **类别质量**：如果初始类别发现有误，所有 PE/PI 更新都会传播错误假设。LLM+规则系统更健壮，因为它每轮重新发现类别。

2. **回答噪声建模**：真实用户的回答比 oracle 嘈杂得多。ε 参数需要仔细校准。

3. **计算成本**：EIG_PE 需要对 L 个类别各 M 个可能回答边际化。对 M=7, L=6, N=24，总计 7×6×24=1008 次运算，可接受。

4. **PI 冷启动**：PI 的 EIG 依赖已确认动作作为证据。贝叶斯框架正确地在无证据时给出 EIG_PI=0。

---

## 12. 参考文献

### 贝叶斯主动学习基础
- Lindley, D.V. (1956). On a Measure of the Information Provided by an Experiment. *Annals of Mathematical Statistics*, 27(4).
- Chaloner, K. & Verdinelli, I. (1995). Bayesian Experimental Design: A Review. *Statistical Science*, 10(3).
- Settles, B. (2009). Active Learning Literature Survey. *CS Tech Report 1648*, UW-Madison.
- Rainforth, T. et al. (2024). Modern Bayesian Experimental Design. *Statistical Science*, 39(1).

### BALD 与信息论采集
- Houlsby, N. et al. (2011). Bayesian Active Learning for Classification and Preference Learning. *arXiv:1112.5745*.
- Kirsch, A. et al. (2019). BatchBALD: Efficient and Diverse Batch Acquisition. *NeurIPS 2019*.
- Melo, L.C. et al. (2024). Deep Bayesian Active Learning for Preference Modeling in LLMs. *NeurIPS 2024*.

### LLM + 贝叶斯偏好引导
- Handa, K. et al. (2024). Bayesian Preference Elicitation with Language Models (OPEN). *arXiv:2403.05534*.
- Boutilier, C. (2002). A POMDP Formulation of Preference Elicitation Problems. *AAAI 2002*.

### 机器人主动偏好学习
- Sadigh, D. et al. (2017). Active Preference-Based Learning of Reward Functions. *RSS 2017*.
- Biyik, E. & Sadigh, D. (2018). Batch Active Preference-Based Learning. *CoRL 2018*.
- Biyik, E. et al. (2019). Asking Easy Questions: A User-Friendly Approach to Active Reward Learning. *CoRL 2019*.
- Biyik, E. et al. (2020). Active Preference-Based GP Regression for Reward Learning. *RSS 2020*.
- Biyik, E. et al. (2022). Learning Reward Functions from Diverse Sources. *IJRR*, 41(1).
- Wilde, N. et al. (2020). Improving User Specifications for Robot Behavior. *IJRR*, 39(6).

### 家居整理
- Wu, J. et al. (2023). TidyBot: Personalized Robot Assistance with LLMs. *IROS 2023*.
- Kapelyukh, I. & Johns, E. (2022). My House, My Rules: Learning Tidying Preferences with GNNs. *CoRL 2021*.
- Wang, H. et al. (2024). APRICOT: Active Preference Learning and Constraint-Aware Task Planning. *CoRL 2024*.
- Ramachandruni, K. & Chernova, S. (2025). Personalized Robotic Object Rearrangement from Scene Context. *ROMAN 2025*.

### 多粒度 / 查询类型选择
- Huang, S.-J. et al. (2015). Multi-Label Active Learning: Query Type Matters. *IJCAI 2015*.
- Li, J. et al. (2018). Cost-Effective Active Learning for Hierarchical Multi-Label Classification. *IJCAI 2018*.

### 对话与会话主动学习
- Su, P.-H. et al. (2016). On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems. *ACL 2016*.
- Abbasnejad, E. et al. (2019). What's to Know? Uncertainty as a Guide to Asking Goal-Oriented Questions. *CVPR 2019*.
- Christiano, P. et al. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS 2017*.

### 序贯设计
- Foster, A. et al. (2021). Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design. *ICML 2021*.
- Russo, D. & Van Roy, B. (2014). Learning to Optimize via Information-Directed Sampling. *NeurIPS 2014*.

### 认知模型
- Chu, W. & Ghahramani, Z. (2005). Preference Learning with Gaussian Processes. *ICML 2005*.
- Jern, A. et al. (2017). People Learn Other People's Preferences Through Inverse Decision-Making. *Cognition*, 168.
