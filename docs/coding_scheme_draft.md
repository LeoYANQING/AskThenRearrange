# Hierarchical Coding Framework — Complete Draft

> 状态：讨论稿，未写入 main.tex
> 版本：整合认识论立场维度（epistemic agency framing）
> 目标：逻辑自洽、学术规范、contribution 清晰

---

## 框架总览

本框架为两层级联结构：

```
原始数据：14 组 HHI 视频 + 转录文本
        ↓
  【Level 1】Intent-level Coding
  对每个 Learner 问题标注意图类别（AO / PE / PI）
  单元：utterance（问题）
  产出：每组对话的意图序列 [Q1, Q2, …, Qn]
        ↓
  【Level 2】Pattern Classification
  对每组对话的意图序列整体分类（DQ / UPF / PAR）
  单元：sequence（对话）
  产出：每组对话的 Pattern 标签 + 分布统计
```

两层分析**线性执行**（Level 1 完成后才进行 Level 2），各自独立报告 Cohen's κ。

---

## Level 1：Intent-level Coding

### 编码单元

Learner 发出的每一个**语义完整的提问话语**（question utterance）。同一回合内多个独立提问拆分计算。

### 三个意图类别

---

#### AO — Action-Oriented

> An **AO query** targets a single, concrete placement or handling decision for a specific item or zone. Its answer resolves one action step; it does not invite the user to articulate any general organizing principle.

**判断标准**：
- 答案的适用范围**仅限于**被问及的单一物品或区域
- Learner 未在问题中提出任何跨物品的布局方案或规则

**认识论角色**：无人构建偏好模型——Learner 只在请求许可或确认，不尝试归纳任何规律

**典型例子**：
- "Should the yogurt go on the top shelf or the bottom?" → AO
- "Can I put the sauce bottle here?" → AO
- "Is this a fresh item?" → AO（事实核查，服务于下一步行动）

---

#### PE — Preference-Eliciting

> A **PE query** invites the user to articulate a general organizing preference or principle. Its answer is expected to apply across multiple items or future decisions. The **user** is the originator of the preference content; the learner is a passive receiver.

**判断标准**：
- 答案能够覆盖**一类物品或整体策略**（泛化范围 ≥ 2 个物品）
- 问题中**不含** Learner 自行构建的任何布局假设或方案

**认识论角色**：**用户**构建偏好框架——Learner 提问、用户表达，Learner 被动接收

**典型例子**：
- "Do you prefer items grouped by type or by frequency of use?" → PE
- "How do you usually organize the drinks section?" → PE
- "Where do you typically put fragile items?" → PE（借助先验类别引导，但内容由用户生成）

**与 AO 的区别**：答案是否有跨物品的泛化力。"Is this fragile?" → AO（事实）；"Where do fragile items go?" → PE（偏好）

---

#### PI — Preference-Induction

> A **PI query** presents a preference hypothesis that the **learner** has inferred from scene observations or accumulated confirmed actions, and asks the user to validate or correct it. The learner is the hypothesis generator; the user is the validator.

**判断标准**：
- 问题中**显式或隐式**包含 Learner 提出的归纳假设（可从场景观察或已有行为推断而来）
- 假设的覆盖范围 ≥ 2 个物品或整体分区

**认识论角色**：**Learner** 主导构建偏好模型——Learner 提出假设，用户验证/修正

**典型例子**：
- "My plan is: milk here, drinks here, fruit here — how does that sound?" → PI（明确方案）
- "So I'll put all open items on the top shelf — does that match your habit?" → PI（从先前行为归纳规则）
- "Everything in this zone seems to be chilled — shall I group all chilled items here?" → PI（从场景观察归纳）

**与 AO 的边界**：
- "Can I put the yogurt here?" → AO（单物品确认，无归纳规则）
- "So everything opened goes on the first shelf, right?" → PI（Learner 自行归纳的规则）；若此规则**尚未被 Learner 归纳、首次询问**→ PE
- "Put this one down here, that one up there — how does that sound?" → **AO**（两个具体物品的捆绑摆放确认，未归纳任何可泛化规则）

**关键边界原则**：PI 要求 Learner 提出的假设是一条**可泛化的偏好规则**——若得到用户确认，该规则应能适用于未被明确提及的其他物品。仅涉及特定物品具体位置的多物品提案，编为 AO（捆绑摆放确认），不构成 PI。

**与 PE 的核心区别**：

| | PE | PI |
|---|---|---|
| 偏好内容的发起者 | 用户 | Learner |
| 问题结构 | 开放式邀请 | 假设 + 验证请求 |
| Learner 的认知角色 | 被动接收 | 主动构建 |

---

### K-type 的辅助作用

K-type（K1–K11，参照 Shin et al. 2023）**不作为独立编码层**，仅作为边界案例的判断辅助：

> To support coding consistency in borderline cases, each question was additionally annotated with its knowledge type (K1–K11; Shin et al. 2023): questions targeting internal states or preferences (K10–K11) were strong candidates for PE or PI, while questions targeting spatial layout or object identity (K1–K7) were strong candidates for AO.

K-type 标注不单独报 κ，不作为独立发现。

### 信度

两名研究者独立对 14 份转录文本中的全部问题进行意图标注，通过讨论解决分歧。整体一致性以 Cohen's κ₁ 报告。

---

## Level 2：Pattern Classification

### 分析单元

每组对话的**完整意图序列**（ordered sequence of intent codes）。

### 认识论立场框架

Pattern 分类的理论基础是：**偏好模型由谁构建？Learner 扮演何种认知角色？**

| Pattern | 偏好结构由谁构建 | Learner 的认识论立场 |
|---|---|---|
| **DQ** | 无人 | 被动执行者（executor）——不尝试构建偏好模型 |
| **UPF** | 用户 | 被动接收者（receiver）——通过 PE 让用户表达，自己接收 |
| **PAR** | Learner（用户验证） | 主动构建者（active constructor）——提出假设，请用户验证 |

三种 Pattern 构成**互斥且完备**的分类（MECE），覆盖所有理论上可能的认识论立场。

### 形式化分类规则

以下三条规则**按优先级顺序应用**：

> **Rule 1.** If any **PI** query appears in the sequence → classify as **PAR**
>
> **Rule 2.** Else if any **PE** query appears **before the first AO** → classify as **UPF**
>
> **Rule 3.** Otherwise → classify as **DQ**

**规则设计的逻辑依据**：

- **Rule 1 优先于 Rule 2**：若序列中同时有 PE（早）和 PI（晚），Learner 兼用了两种策略；但 PI 的存在代表 Learner 曾主动生成偏好假设，认识论立场更复杂，整体以 PAR 定性。
- **"before the first AO" 而非"前 20%"**：UPF 的语义是"preference first"——PE 必须在第一个行动决策（AO）之前提出，才构成真正的 top-down 框架优先策略。"before first AO"是语义精确的表达，"前 20%"是近似值。
- **Rule 3 不需要正向判断**：DQ 是剩余类（residual category）——没有任何偏好求取行为，Learner 从头到尾以 AO 执行，不尝试建模。

### 边界案例处理

**PE 出现在第一个 AO 之后**（如：AO → AO → PE → AO）：
- 无 PI，且 PE 不在第一个 AO 之前 → 归 DQ
- 理由：PE 出现过晚，Learner 已在无偏好框架的状态下开始执行，不符合"preference first"的语义

**PI 出现在对话第一问**（如：PI → AO → AO）：
- 有 PI → 归 PAR
- 理由：PI 基于场景观察归纳假设，不要求必须有先行 AO；第一问即 PI 说明 Learner 从一开始就采取主动构建立场

### 信度

两名研究者独立对 14 个意图序列进行 Pattern 分类，通过讨论解决分歧。整体一致性以 Cohen's κ₂ 报告。

---

## 框架自洽性检验

以下是框架的四个内部一致性命题，均可从定义直接推导：

1. **AO 与 PE/PI 不重叠**：AO 的答案范围 = 单一物品；PE/PI 的泛化范围 ≥ 2 个物品。互斥。

2. **PE 与 PI 的主体互斥**：PE 中偏好内容由用户生成；PI 中偏好假设由 Learner 生成。同一问题不可能同时满足。

3. **DQ/UPF/PAR 完备**：Rule 1 + Rule 2 + Rule 3（residual）覆盖所有可能序列。

4. **PAR 优先于 UPF 的认识论理由**：PAR 代表 Learner 具有更高的认知主体性（提出假设），UPF 代表 Learner 是被动接收者。若两者共存，Learner 的认识论立场以更高的主体性定性。

---

## 与 §4 系统设计的接口

HHI 中观察到的三种 Pattern 对应三种**系统可实现的提问策略**：

| HHI Pattern | 系统策略 | 映射说明 |
|---|---|---|
| DQ | DQ strategy | 仅 AO 问题，逐项询问 |
| UPF | UPF strategy | 对话开始时优先 PE，再执行 AO |
| PAR | PAR strategy | 交替 AO 与 PI，持续主动归纳假设 |

**注**：HHI 中的 PI 对应系统中的 PI 问题类型，但系统实现中 PI 的假设来源是系统的推理模块（而非人类的直觉归纳）。

---

## 待确认事项（写入 main.tex 前需解决）

| 问题 | 当前方案 | 状态 |
|---|---|---|
| K-type 位置 | 降级为辅助说明，不独立报 κ | ✅ 确认 |
| PI 定义中"场景观察"是否纳入 | 纳入（与第一问 PI 相容） | ✅ 确认 |
| Pattern 认识论立场维度 | 纳入 Level 2 定义 | ✅ 确认 |
| GH010062 Q11 归类 | 重编为 AO（捆绑摆放确认，非可泛化规则），GH010062 = UPF | ✅ 确认 |
| κ₁ / κ₂ 具体数值 | TODO | ⏳ 待填 |
| 分布数字（DQ=3, UPF=10, PAR=1） | ✅ 确认最终数字 | ✅ 确认 |
| §3.3 引言段是否保留"hierarchical" | 保留（两层结构，术语准确） | ✅ 确认 |
| 归纳方法标签 | "iterative analysis"，不引用 Braun & Clarke | ✅ 确认 |
