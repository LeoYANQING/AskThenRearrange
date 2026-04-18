# 写作逻辑指南 — AskThenRearrange / PrefQuest

> 本文件规范论文的**叙事逻辑和论证结构**，与 `writing_style_guide.md`（格式/统计/术语规范）互补。  
> 每次写作或修改 main.tex 之前必须与风格指南一起参照。

---

## 一、论文整体论证链

本论文的核心论证由三个递进命题组成，每个命题对应一个主要贡献：

```
命题 1（HHI Study，§3）
人类在教授家居整理任务时，自然呈现三种可辨别的提问模式。
→ 这确立了三种策略的经验来源，赋予其生态效度。

        ↓ 因此值得形式化

命题 2（System Design，§4）
这三种模式可以被形式化为可组合的问题元素（AO/PE/PI）
并在结构化偏好学习框架（PrefQuest）中实例化为可控策略。
→ 这确立了技术可行性，并使命题3的对比实验成为可能。

        ↓ 既然可控，就应该测量差异

命题 3（Study 1，§5）
策略选择对目标任务绩效产生可量化的显著差异，
尤其体现在未见物品的偏好满足率上。
→ 这是论文的主要实证贡献，回答 RQ3。

        ↓ 但这是在模拟条件下——

命题 4（Study 2，§6，待完成）
上述差异在真实用户交互中仍然成立，
且不同策略在主观体验维度上产生不同的权衡。
→ 这确立生态效度，回答 Study 2 的 RQ1–RQ3。
```

**写作时的核心要求**：每个章节的论证必须在这条链上找到自己的位置，并明确交代它如何为下一个命题铺垫。不能孤立地叙述某一节的内容。

---

## 二、Introduction 的论证逻辑

Introduction 的任务是让读者在读完之后认为："这个问题重要、这个 gap 真实存在、这篇论文做了正确的事情。"

### 漏斗结构（必须遵守）

```
段落 1-2：宏观动机
  家居机器人需要偏好 → 偏好是个人化的 → LBA 是获取偏好的自然机制

段落 3：Gap（关键段落）
  现有 LBA 研究聚焦于"问什么"（原子问题优化），
  忽视了"按什么顺序问"（宏观策略组织）。
  现有评估几乎只用主观指标，缺乏客观任务绩效证据。

段落 4：本文方案
  RQ1/RQ2/RQ3 明确编号，逐条对应后续章节

段落 5：主要发现（结果预告）
  不在 Abstract 里说完的发现，在 Introduction 末尾再点一次

段落 6：论文结构（roadmap）
  "The remainder of this paper is organized as follows..."
```

### 常见逻辑错误

**错误**：先列贡献，再说 gap。  
**正确**：gap 必须先于贡献出现，贡献是 gap 的答案，顺序不能颠倒。

**错误**：RQ 写得太宽泛（"How do strategies affect users?"）。  
**正确**：RQ 必须可以被具体的数据直接回答（"Does UPF produce higher unseen PSR than DQ at B=5?"）。

**错误**：Introduction 就提实验数字（85.2%）。  
**正确**：Introduction 只提方向性结论，具体数字留给 Results 和 Abstract。

---

## 三、Related Work 的论证逻辑

Related Work 的任务不是展示文献知识，而是**构建 gap 的证据链**。每个子节都在回答：为什么现有工作没有解决我们的问题？

### 每个子节的标准结构

```
1. 开篇定义子领域（1句）
2. 综述主流发现——聚合写法，不要逐篇列举（2-3句）
3. 指出这个子领域的局限（1-2句）
4. 收尾句：本文与之的区别（1句，必须写）
```

### 收尾句的标准写法

> "The present work extends this line of research by [具体区别], providing the first [量化评估/对比实验/形式化框架]."

不能只写"The present work is related to these efforts"——这是废话，没有定位。

### 四个子节的论证分工

| 子节 | 建立的前提 |
|---|---|
| §2.1 Household Robot Learning | 家居场景的偏好是个人化的，不能预编码 |
| §2.2 Learning by Asking | LBA 是有效机制，但现有研究都是原子问题优化 |
| §2.3 Dialogue & Questioning Strategies | 问题序列的宏观组织影响信息获取效率 |
| §2.4 LLMs for Robot Learning | 结构化状态比原始对话历史更可靠 |

每个子节建立的前提，在 Introduction 的 gap 段落里都应该有对应的一句话引用。Related Work 和 Introduction 的 gap 论证必须相互呼应。

---

## 四、Results 的论证逻辑

Results 的任务是**用数据直接回答 RQ**，而不是汇报所有统计检验。

### 组织原则：按 RQ 而非按统计检验

```
§5.2.1 标题 = RQ 的简化版
  → 第一句重述本节回答的 RQ
  → 报告主要发现的方向和量级（M ± SE）
  → 报告检验统计量（W/F/t，p，effect size）
  → 解释发现的方向（"This advantage emerged because..."）
  → 报告非显著发现（必须，不能省略）
```

### 本论文 Results 的三层逻辑

**第一层：策略之间有差异吗？**（主要发现）  
UPF > DQ on unseen PSR，差异显著且量级大（+12.5 pp）。

**第二层：为什么 PAR 没有优势？**（反直觉发现，必须解释机制）  
PAR 使用了 PI（归纳元素），理论上应该泛化，但在 B≤5 的预算下，PAR 必须先花 AO turns 积累证据才能触发 PI，导致剩余预算不足。这是**结构性原因**，不是随机噪声。

> "This finding appears structural: within a budget of five questions, PAR must first spend at least two AO turns at the same receptacle before PI can be triggered, leaving fewer remaining turns for rule-driven generalization."

**第三层：HYB 为什么不优于 UPF？**（自适应组合的理论预期 vs 实证结果）  
PE 在有未覆盖 receptacle 时始终优先被选择，这个条件在典型 episode 中贯穿整个预算，导致 HYB 在实践中退化为 UPF。

> "The theoretical benefit of combining PE and PI is therefore not realized in practice under the tested constraints."

**写作要点**：这三层发现必须都有机制解释，不能只说"PAR 表现差于 UPF，p < .001"而不说为什么。IJHCS 审稿人要求结果可解释。

---

## 五、Discussion 的论证逻辑

Discussion 的任务是**将数字还原为意义**，并指向实践。它不是 Results 的复述，而是 Results 的升华。

### §5.1 Interpretation 的写法逻辑

每段的结构：
```
1. 用一句话引出本段要解释的现象（指向具体数字）
2. 提出机制解释（"This pattern arises because..."）
3. 用已有文献支撑这个机制（如果有相关文献）
4. 说明这个机制的边界条件（什么时候成立，什么时候不成立）
```

不能只说"UPF 表现更好，这与 H1 一致"——必须解释为什么 PE questions 能产生可泛化的规则，而 AO questions 不能。

### §5.2 Design Implications 的写法逻辑

每条 Implication 的结构：
```
粗体标题（直接说给设计者的建议）
→ 数据支撑（"At B=5, UPF achieved... suggesting that..."）
→ 适用条件（什么样的系统/场景适用这条建议）
→ 如果有反例或边界，主动说明
```

Implications 必须是**可操作的**，不能是泛泛的。

**错误**："Designers should consider using better questioning strategies."  
**正确**："When the interaction budget is constrained to B ≤ 5, agents should initiate dialogue with category-level preference elicitation rather than object-specific queries, as a single PE question can cover the placement logic for multiple unseen objects."

### §5.3 Bridging Simulation and Real Interaction

这一节的逻辑功能是：承认模拟实验的局限，并将其转化为对 Study 2 的动机铺垫。写法：

```
1. 肯定 Study 1 的发现（"The present findings provide clear evidence that..."）
2. 指出 oracle 局限（"A key limitation is the use of an LLM-based oracle..."）
3. 提出 Study 2 作为回应（"Section 6 describes a user study designed to address this question directly."）
4. 预告两种可能结果的理论意义（"If the simulation findings hold... If they do not..."）
```

不能只说"Study 2 会验证"——必须说清楚不管结果如何都有意义。

### §5.5 Limitations 的写法逻辑

Limitations 必须**诚实而具体**，不能用"未来工作可以拓展"代替真正的局限承认。

三个必须覆盖的局限（已在论文中）：
1. **Simulated oracle**：LLM oracle 不能完全模拟人类回答的变异性
2. **Single domain**：只在家居整理场景验证，泛化到其他领域未知
3. **Model dependence**：绝对 PSR 值随 LLM backbone 变化（但相对排序稳定）

每个局限的写法：`局限是什么 → 它可能影响什么结论 → 已有什么缓解措施`

---

## 六、Study 1 与 Study 2 的叙事衔接逻辑

这是本论文最重要的叙事关节，处理不好会让审稿人觉得两个研究是拼凑的。

### 正确的衔接逻辑

Study 1 和 Study 2 的关系是**同一论证的两个阶段**，而不是两篇独立研究：

```
Study 1 建立了什么：
  在可控条件下，策略差异对客观绩效的因果效应
  （排除了其他变量，只有策略不同）

Study 1 留下了什么问题：
  模拟 oracle 是否能代表真实用户？
  客观绩效优势是否伴随主观体验优势？

Study 2 回答了什么：
  RQ1：效应是否在真实用户条件下复现？
  RQ2：策略如何影响主观体验（认知负荷/满意度/感知控制）？
  RQ3：客观绩效与主观体验是否存在解离？
```

### 衔接的写作位置

- **Study 1 Results 末尾**：不是总结，而是提出"open question"（oracle limitation）
- **Study 2 Section 开篇**（已写入 §6）："Study 1 established that... An important open question is whether..."
- **Discussion §5.3**：用 Study 1 的数字作为"预期基准"，为 Study 2 结果的解释提供参照

---

## 七、Study 2 Results 的论证逻辑（待写，数据收集后参照）

Study 2 Results 需要同时回答三个 RQ，叙事结构如下：

### §6.7 Objective Performance（回答 RQ1）

核心论证任务：判断模拟效应是否在真实用户中复现。

写法逻辑：
```
1. 报告三种策略的 unseen PSR（M ± SE），配表格
2. 统计检验（Friedman + Wilcoxon，或 ANOVA）
3. 与 Study 1 的数字做对比（"This gap is smaller than the +12.5 pp observed in Study 1..."）
4. 如果效应缩小：解释为实用户变异性带来的衰减，而不是"结果不一致"
5. 如果效应消失：诚实报告，并探讨可能原因（样本量、任务设计、oracle 假设的特殊性）
```

**关键 hedge**：永远不要说"Study 1 的发现被 Study 2 证实"——说"Study 2 的结果与 Study 1 的方向一致"或"部分复现了模拟研究的发现"。

### §6.8 Questionnaire Results（回答 RQ2）

核心论证任务：不同策略产生不同的主观体验模式。

潜在的有趣发现模式：
- **一致**：UPF 客观最优 + 主观满意度最高 → 简单结论，无需太多解释
- **解离**：UPF 客观最优 + 认知负荷更高（因为 PE questions 要求更多抽象思考） → 重要发现，说明不同维度优化不同策略
- **反转**：DQ 主观满意度高（因为问题具体易回答）但客观绩效低 → 说明 accuracy-usability trade-off 存在

写法：无论哪种模式，都要在 Discussion 中用 §十三.5 的主客观解离句型处理。

### §6.9 Interview Results（回答 RQ3 的质性部分）

主题分析结果的写法：
```
1. 说明主题数量和 κ 值
2. 逐主题报告（主题名 → 代表性引用 → 跨条件比较）
3. 每个主题必须与客观/主观数据产生对话（"This perception is consistent with..."）
```

---

## 八、常见论证错误清单

| 错误类型 | 错误示例 | 正确做法 |
|---|---|---|
| 只报数字不解释机制 | "UPF outperformed DQ (p < .001)" | 加一句"because PE questions yield category-level rules that generalize to multiple unseen objects" |
| 省略非显著结果 | 只写显著的比较 | PAR vs DQ 的非显著差异必须报告并解释 |
| 用 Study 2 "证明" Study 1 | "Study 2 confirmed our simulation findings" | "Study 2 results were consistent with / partially replicated..." |
| Related Work 无定位句 | 子节只综述文献 | 每个子节末尾必须有 "The present work differs from..." |
| Implications 流于泛泛 | "Designers should ask better questions" | 给出具体条件和操作（B ≤ 5时，先问PE，再问AO边界案例） |
| 过度陈述 | "This proves that top-down is better" | "These results suggest that top-down preference elicitation appears more efficient under tight budgets" |
| 论证顺序错误 | Introduction 先写贡献再写 gap | Gap → 贡献，顺序不可颠倒 |
| 跨研究数字混用 | 用 Qwen3 数字（69.8%）写主文 | 主文只用 GPT-5-Chat 数字（85.2%） |

---

*最后更新：2026-04-18。Study 2 数据收集后，在§七补充实际结果的论证模板。*
