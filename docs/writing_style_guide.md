# IJHCS 写作风格指南

> 本文档基于以下5篇 IJHCS（International Journal of Human-Computer Studies）代表性论文，抽象出写作惯例与风格规范，供润色、写作时参照。

---

## 参考论文来源

| # | 标题 | 年份 | DOI |
|---|------|------|-----|
| P1 | The effect of preference elicitation methods on the user experience in conversational recommender systems | 2024 | 10.1016/j.ijhcs.2024 (pii S0885230824000792) |
| P2 | Understanding the user experience of customer service chatbots: An experimental study of chatbot interaction design | 2022 | 10.1016/j.ijhcs.2022.102788 |
| P3 | Can robot advisers encourage honesty?: Considering the impact of rule, identity, and role-based moral advice | 2024 | 10.1016/j.ijhcs.2024.103217 |
| P4 | ContractMind: Trust-calibration interaction design for AI contract review tools | 2024 | 10.1016/j.ijhcs.2024.103411 |
| P5 | Empathy in action: An empirical exploration of user perspectives on conversational agent empathy | 2026 | 10.1016/j.ijhcs.2026.103797 |

---

## 一、论文整体结构

IJHCS 的实证研究论文遵循严格的 IMRaD 变体结构：

```
Abstract（~200–250词）
1. Introduction
2. Related Work（含2–4个子节）
3. Method / Study Design
4. Results
5. Discussion
   5.1 Interpretation of Findings
   5.2 Design Implications（专用子节，IJHCS 标志性特征）
   5.3 Limitations and Future Work
6. Conclusion
References
```

**特别注意**：Design Implications 是 IJHCS 强调的独立子节，不能省略或与 Discussion 合并写。须将研究发现直接转化为对系统设计者的可行性建议，用"我们建议设计者..."或"These findings suggest that systems should..."句式落地。

---

## 二、Abstract 写法

IJHCS 的 Abstract 呈隐式结构化：背景→问题→方法→结果→意义，约200–250词，不使用子标题，但信息密度极高。

**模板示例（对标 AskThenRearrange）**：
> Intelligent agents capable of learning user preferences through dialogue hold promise for personalizing household rearrangement tasks. However, it remains unclear how different question-asking strategies affect an agent's ability to generalize learned preferences to unseen objects. This paper presents a systematic comparison of four questioning strategies—Direct Querying, User-Preference-First, Parallel Exploration, and Hybrid-All—evaluated across 102 household episodes under varying question budgets. Results indicate that the User-Preference-First strategy significantly outperforms Direct Querying in unseen object accuracy (85.2% vs. 72.7% at budget B=5), suggesting that eliciting abstract preference rules enables broader generalization than asking about individual items. These findings have implications for the design of preference-aware dialogue agents that must operate under interaction budget constraints.

> *(数字来源：GPT-5-Chat 全量实验，n=102 episodes。本模板仅为格式示例；如 Abstract 正文数字有更新，以 main.tex 为准。)*

**关键规范**：
- 直接说明 N（样本/episode数），不说"a large dataset"
- 结果用具体数字，不用"better performance"
- 末句必须有 implication（"These findings suggest..."或"This has implications for..."）

---

## 三、Introduction 写法

IJHCS Introduction 遵循"漏斗式"三段逻辑：

**段落1 — 宏观动机（1–2段）**：
从领域背景切入，确立研究的社会或技术重要性。常用短语：
- "X has gained increasing attention in recent years..."
- "As AI-powered systems become increasingly integrated into everyday life, ..."
- "The deployment of conversational agents for Y represents a promising direction for..."

**段落2 — 研究空白（1段，关键）**：
精准指出现有文献的局限，用对比句型强调 gap：
- "However, most prior work has focused on X, leaving Y underexplored."
- "Despite significant advances in X, it remains unclear whether/how Y..."
- "While [Author, Year] demonstrated Z, the question of Q has not been addressed."

**段落3 — 本文贡献（1段）**：
逐条列出贡献，常见句型：
- "To address this gap, this paper makes the following contributions: (1)...(2)...(3)..."
- "The present study investigates..."
- "We report the results of a controlled experiment examining..."

**段落4 — 论文结构（1段，可选）**：
"The remainder of this paper is organized as follows: Section 2 reviews..."

---

## 四、Research Questions / Hypotheses 的写法

IJHCS 偏好将 RQ 或 H 明确编号，置于 Introduction 末尾或 Related Work 末尾，形成可追溯的锚点。

**RQ 格式（适用于探索性研究）**：
> **RQ1**: Does the type of questioning strategy (Direct Querying vs. User-Preference-First) affect unseen object placement accuracy?
> **RQ2**: How does question budget interact with strategy type to influence generalization performance?

**H 格式（适用于有理论预期的研究）**：
> **H1**: User-Preference-First will yield higher unseen accuracy than Direct Querying at the same question budget, owing to the broader coverage of abstract preference rules.
> **H2**: The accuracy advantage of User-Preference-First over Direct Querying will increase as budget decreases, as the value of each question is higher under constraint.

**注意**：
- 研究问题的主语通常是"the strategy" / "question type"而不是"we"
- 假设的方向必须有文献依据，通常在 Related Work 中铺垫

---

## 五、Related Work 写法

IJHCS 的 Related Work 通常分2–4个子节，每节先综述主流观点再指出局限。

**子节标题示例（对标 AskThenRearrange）**：
```
2.1 Preference Elicitation in Conversational Systems
2.2 Active Questioning Strategies for User Modeling
2.3 Generalization of Learned Preferences to Unseen Items
```

**段落句式规律**：
- 开篇定义概念："Preference elicitation refers to the process by which a system gathers information about user preferences, typically through direct questioning or behavioral observation (Author, Year)."
- 引述多篇文献时用综述句："A number of studies have examined X (Author1, Year1; Author2, Year2; Author3, Year3)."
- 对比研究："While [Author, Year] focused on X, [Author, Year] examined Y."
- 子节收尾必须回归本文："The present work differs from prior efforts in that we explicitly compare..."

**写作忌讳**：
- 不要逐篇介绍（"Smith did X, Jones did Y, Lee did Z..."），要按主题聚合
- 不要只写"别人研究了什么"，要写"他们发现了什么，还缺什么"

---

## 六、Method 写法

IJHCS 实验设计叙述要求精准、可复现，按以下顺序组织：

### 6.1 研究设计（Design）
明确说明 between-subjects / within-subjects / mixed design，以及自变量数量与水平：
> "We employed a 4×5 between-subjects design, with questioning strategy (Direct Querying, User-Preference-First, Parallel Exploration, Hybrid-All) and question budget (B = 1, 3, 5, 7, 10) as independent variables."

### 6.2 参与者 / 数据集（Participants / Dataset）
- 对用户研究：报告 n、性别比例、年龄 M(SD)、招募方式
- 对模拟实验：报告 episode 数量、数据集来源、场景多样性
> "Experiments were conducted on a dataset of 102 household episodes, each comprising [description]. Episodes were sampled to ensure diversity across room types and object categories."

### 6.3 实验流程（Procedure）
按时间顺序叙述，用过去时：
> "Participants were first presented with a description of the household environment. The agent then posed up to B questions, following one of four pre-assigned strategies. Upon completion, the agent predicted the placement location for all objects."

### 6.4 测量指标（Measures）
每个 metric 给一个短定义，公式或计算方式说清楚：
> "We report **seen accuracy** (the proportion of seen objects correctly placed) and **unseen accuracy** (the proportion of unseen objects correctly placed) as primary outcome measures."

### 6.5 分析方法（Analysis）
> "Data were analyzed using repeated-measures ANOVA with strategy as a between-subjects factor and budget as a within-subjects factor. Post-hoc pairwise comparisons were conducted using Tukey's HSD correction."

---

## 七、Results 写法

**结构原则**：按 RQ 或 H 组织，而非按统计检验类型。

**标准段落结构**：
1. 重述本段回答的 RQ
2. 描述描述统计（M, SD）
3. 报告推断统计（含 effect size）
4. 解释方向

**统计报告格式（必须遵守）**：
```
显著结果：F(1, 98) = 12.43, p = .001, η² = .11
非显著结果：F(1, 98) = 0.82, p = .367
t检验：t(45) = 3.21, p = .002, d = 0.94
百分比差异：User-Preference-First (M = 69.8%, SD = 8.2%) significantly outperformed Direct Querying (M = 58.7%, SD = 9.1%)
```

**常用句型**：
- "Results indicate that strategy type had a significant main effect on unseen accuracy, F(3, 404) = 14.22, p < .001, η² = .10."
- "Pairwise comparisons revealed that User-Preference-First significantly outperformed Direct Querying (p = .003, d = 0.71)."
- "No significant difference was observed between Hybrid-All and User-Preference-First (p = .812), suggesting structural equivalence between the two strategies."
- "As shown in Fig. 2, unseen accuracy increased monotonically with budget for User-Preference-First but plateaued after B = 6 for Direct Querying."

**注意**：非显著结果必须报告，不能忽略。负面结果也是发现。

---

## 八、Discussion 写法

IJHCS Discussion 的核心是"将数字还原为意义"，有三个必要动作：

### 8.1 重述发现并解释（Interpretation）
每段以一句"This study examined..."或"The results of this study..."开头，然后用先前文献解释为什么结果如此：
> "The superiority of User-Preference-First over Direct Querying in unseen accuracy aligns with findings from [Author, Year], who showed that abstract rule extraction generalizes more broadly than item-level associations. This suggests that..."

### 8.2 Design Implications（设计启示，专用子节）
**这是 IJHCS 的核心贡献载体**，须给出具体、可行的设计建议：
> "Based on our findings, we suggest the following design implications for developers of preference-aware dialogue agents:
> **Implication 1 – Prioritize rule-level questioning under tight budgets.** When the number of available interactions is limited (B ≤ 5), agents should ask abstract category-level questions before item-level questions, as this maximizes generalization to objects the agent has not yet encountered.
> **Implication 2 – Avoid clustering-free item queries early in the dialogue.** Querying unrelated objects in sequence yields insufficient evidence for preference induction and reduces the effectiveness of subsequent rule-generalization strategies."

### 8.3 Limitations（局限性）
IJHCS 对 Limitations 要求诚实而具体，至少覆盖：
- 样本/场景代表性："The current study was limited to household scenarios with a fixed set of receptacles and object categories; generalizability to other domains remains to be demonstrated."
- 用户模拟问题（如果使用 LLM 作为 oracle）："User responses were simulated using a large language model rather than collected from real users, which may not fully reflect human variability in preference expression."
- 外部效度："All episodes were conducted in simulation; the extent to which findings transfer to embodied, real-time interaction settings is an open question."

---

## 九、语气与措辞规范

### 9.1 人称使用
- **我们（we）**：用于"我们做了什么"的主动描述（"We conducted", "We recruited", "We designed"）
- **被动语态**：用于方法和程序的中性描述（"Participants were assigned to", "Data were collected using"）
- **第三人称**：用于引述自己论文时（"The present study", "The current work", "This paper"）

### 9.2 程度限制语（Hedging）

IJHCS 明确反对过度陈述（overclaiming），必须用程度限制语：

| 强陈述（避免） | 正确的 hedged 表达 |
|---|---|
| "proves that" | "provides evidence that" / "suggests" |
| "X causes Y" | "X is associated with Y" / "X appears to influence Y" |
| "users prefer X" | "participants in this study rated X higher" |
| "our system is better" | "our approach outperformed the baseline in terms of unseen accuracy" |
| "the results show" (接绝对结论) | "results indicate" / "findings are consistent with" |

### 9.3 数字与单位
- 小于10的整数写出英文单词（"four strategies"），大于等于10写数字（"102 episodes"）
- 百分比：保留一位小数（"69.8%"），不写"~70%"
- p 值：p = .001（不带0前缀），p < .001（而非 p < 0.001）
- 统计量：F、t、p 斜体，η²、d 亦斜体

### 9.4 常用连接语

**引入文献**：
- "Prior work has shown that..." / "Previous research suggests..."
- "Building on [Author, Year], we..." / "Extending the framework of..."
- "In contrast to [Author, Year], who found..."

**引入结果**：
- "Consistent with H1, results indicate..." / "Contrary to our expectations, ..."
- "As hypothesized, ..." / "Unexpectedly, ..."

**引入讨论**：
- "Taken together, these findings suggest..."
- "One possible explanation for this pattern is..."
- "This effect may reflect the fact that..."

**引入启示**：
- "These findings have implications for the design of..."
- "Practitioners seeking to build X should consider..."
- "This suggests that future systems should..."

---

## 十、专用于 AskThenRearrange 论文的写作规范

### 10.1 核心术语的一致性

以下术语须在全文保持完全一致，首次出现时给出定义：

| 概念 | 标准写法 | 避免写法 |
|---|---|---|
| 四种策略 | Direct Querying (DQ), User-Preference-First (UPF), Parallel Exploration (PAR), Hybrid-All (HYB) | "the first strategy" / "our method" / "PE" for strategy / "HA" for Hybrid |
| 三种问题元素 | Action-Oriented (AO), Preference-Eliciting (PE), Preference-Induction (PI) | "AO question" / "pattern AO" / "Preference Eliciting" without hyphen |
| 指标 | seen PSR, unseen PSR（Preference Satisfaction Rate） | "seen acc." / "seen accuracy" / "accuracy for seen objects" |
| 预算 | question budget $B$ | "question limit" / "dialogue turns" / "budget B" without math mode |

> ✅ **命名冲突已解决**（v1.1 更新）：原草稿中 PE 同时用于"Parallel Exploration"策略和"Preference-Eliciting"问题元素，造成歧义。当前定稿方案：策略一律用 **PAR**（Parallel Exploration），问题元素一律用 **PE**（Preference-Eliciting）。全文不得再出现将 Parallel Exploration 缩写为 PE 的用法。Hybrid-All 统一缩写为 **HYB**，不使用 HA。

### 10.2 如何描述 UPF ≡ Hybrid 的发现

这是一个反直觉但有价值的发现，写法参考 P3（三项研究结果均非显著的诚实报告）：
> "Notably, Hybrid-All and User-Preference-First produced statistically indistinguishable results across all budget levels (p > .05 for all pairwise comparisons). This structural equivalence arises because the Preference-Eliciting pattern dominates strategy selection whenever uncovered objects remain—a condition that holds throughout the question budget in our scenario. This finding suggests that the theoretical benefit of combining PE and PI is not realized in practice under the constraints studied here, and has implications for how multi-strategy systems should be evaluated and reported."

### 10.3 LLM-as-oracle 的标准写法

IJHCS 审稿人对 LLM 模拟用户会有疑虑，需主动说明并 hedge：
> "In this study, user responses were generated by [Model Name], a large language model, acting as a simulated oracle. While this approach enables large-scale, reproducible experimentation, it does not capture the full variability of human responses. We treat the results as indicative of relative strategy performance rather than absolute accuracy in human-agent dialogue."

### 10.4 Study 1（模拟实验）与 Study 2（用户研究）的衔接

参考 P5（Empathy in action 的两阶段结构）：
> "Study 1 established the objective performance differences between strategies at scale (N = 102 episodes). Study 2 extends this by examining user experience: do strategies that achieve higher placement accuracy also produce more satisfying interactions from the user's perspective?"

---

## 十一、Figure 和 Table 的规范

**Figure 标题格式**：
> "Fig. 1. Unseen accuracy as a function of question budget for each questioning strategy. Error bars represent ±1 SE."

**Table 标题格式**（置于表格上方）：
> "Table 1. Mean (SD) seen and unseen accuracy across questioning strategies at B = 5."

**正文引用图表的句型**：
- "As illustrated in Fig. 2, UPF consistently outperforms DQ across all budgets."
- "Table 2 summarizes the pairwise comparison results."
- "The interaction pattern (see Fig. 3) indicates that..."

**不要写**：
- "The following figure shows..." （直接说内容）
- "See table 1 for details." （先说结论再引表）

---

## 十二、常见被拒原因（写作角度）

根据 IJHCS 审稿惯例，以下写作问题最易导致 major revision 或 rejection：

1. **RQ 不明确**：Introduction 中没有明确编号的 RQ 或 H，Discussion 中无法追溯
2. **Design Implications 缺失或流于泛泛**：只写"future systems should consider user preferences"之类废话
3. **Limitation 不足**：未讨论 LLM oracle 的局限、样本多样性、外部效度
4. **过度陈述**：用"proves" / "demonstrates that AI is better than humans"等绝对性语言
5. **统计报告不规范**：缺 effect size、不报 non-significant results、p 值格式错误
6. **Related Work 未定位**：每个子节末尾未回到"本文与前人工作的区别"
7. **摘要与结论不一致**：Abstract 中的主要结论与 Conclusion section 措辞差异明显

---

---

## 十三、User Study 写作专项规范

本节针对 Study 2（用户实验）的写作场景，补充§六 Method 通用规范之外的具体要求。

### 13.1 Post-dialogue Preference Form 的标准表述

这是 Study 2 方法设计的核心决策，必须主动与"pre-registered gold label"区分，否则审稿人会质疑 ground truth 的独立性。

**正确写法**：
> "The post-dialogue preference form served as the participant's reference preference annotation for that trial, reflecting preferences as articulated following the teaching interaction and independent of any system output."

> "In the paper, this elicitation step is described as a post-interaction ground-truth measure rather than a pre-registered gold label, consistent with the within-subjects teaching paradigm."

**避免写法**：
- ~~"ground truth preferences"~~（暗示预存在、客观不变）
- ~~"the correct placements"~~（价值判断）
- ~~"pre-defined user preferences"~~（与实际收集时机矛盾）

### 13.2 被试内设计的统计结论措辞

Study 2 使用 within-subjects 设计，统计检验为 Friedman + Wilcoxon signed-rank（若非正态）或 repeated-measures ANOVA（若正态）。

**Friedman 检验报告格式**：
> "A Friedman test revealed a significant effect of strategy on unseen PSR, $\chi^2(2) = 8.14$, $p = .017$."

**Wilcoxon signed-rank 事后比较格式**（配合 Holm correction）：
> "Post-hoc Wilcoxon signed-rank tests with Holm correction indicated that UPF produced significantly higher unseen PSR than DQ ($W = 47$, $p = .023$, $r = .52$)."

**repeated-measures ANOVA 格式**：
> "A one-way repeated-measures ANOVA with Greenhouse–Geisser correction revealed a significant main effect of strategy, $F(1.74,\ 39.9) = 6.83$, $p = .004$, $\eta^2_p = .23$."

**Hedging 程度**：Study 2 结果中，若方向与 Study 1 一致，可用"replicates"或"is consistent with"；若效应量缩小，须主动说明并 hedge：
> "The advantage of UPF over DQ on unseen PSR was smaller in magnitude than in Study 1 ($+$5.3~pp vs.\ $+$12.5~pp), consistent with the attenuation expected when real-user response variability is introduced."

若结果未达显著，不能回避，须诚实报告并提供可能解释：
> "No significant difference was observed between UPF and DQ on unseen PSR ($p = .182$), which may reflect the reduced effect size under real-user variability or insufficient power given the conservative sample-size target."

### 13.3 Study 1 与 Study 2 PSR 数值的跨实验比较

直接对比两个研究的绝对 PSR 数值时，必须指明实验条件差异，避免误导性比较：

**正确写法**：
> "Under simulation (Study 1, GPT-5-Chat, $n = 102$~episodes), UPF achieved $85.2\%$ unseen PSR at $B = 5$. In the user study (Study 2, $N = 24$~participants), the corresponding figure was $M = $~\textbf{[TODO after data collection]}\%, suggesting [qualitative direction of replication]."

**避免写法**：
- ~~"Study 2 replicated the 85.2% result"~~（除非数字相同）
- ~~"performance dropped from Study 1 to Study 2"~~（应先说明两者条件差异）

### 13.4 策略偏好排名（非参数比较）的报告格式

Study 2 收集策略偏好排名（1st/2nd/3rd），须用非参数检验报告：

> "Strategy preference rankings were analysed using a Friedman test. [Result]. Post-hoc pairwise comparisons using Wilcoxon signed-rank tests with Holm correction indicated that [specific pairwise differences]."

若报告各策略被选为第一名的频率：
> "UPF was ranked first by $n = 11$ of 24 participants (45.8%), followed by PAR ($n = 8$, 33.3%) and DQ ($n = 5$, 20.8%)."

### 13.5 主客观解离的标准表述

若出现高 PSR + 低满意度（或反之）的解离模式，参考以下写法：
> "A notable dissociation emerged between objective placement accuracy and subjective satisfaction: although UPF produced the highest unseen PSR ($M = $~\%), participants did not rate it as the most preferred strategy ($p = .xx$). This pattern suggests that users may prioritize interaction naturalness or perceived control over task performance per se, consistent with prior findings on the 'accuracy–usability trade-off' in recommender systems \citep{knijnenburg2012explaining}."

---

*最后更新：2026-04-18。如有新的参考论文或风格发现，请在此文档追加。*
