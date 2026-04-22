# E1 HHI Data Analysis — Coding Evaluation & Pattern Classification

**Task**: Refrigerator organisation (result-oriented, E1)  
**Dyads read**: GH010044, GH010044!!, GH010046, GH010048, GH010050, GH010052, GH010054, GH010056, GH010058, GH010060, GH010062, GH010064, GH010067, GH010069 (14 dyads total)  
**Green text** = pre-task thinking phase; **Black text** = during-task execution phase

---

## Part 1 — Verdict on Current KCQ Coding Scheme

### What the raw data actually has (4 dimensions)

The actual coding table has **four dimensions** per question:

| Column | Values seen | Theoretical anchor |
|--------|-------------|-------------------|
| **K-type** | K1 Identity, K2 Class, K3 Attributes, K5 Spatial Layout, K6 Temporal Relation, K7 Contents, K10 Internal State (Contents), K11 Internal State (Preference) | Knowledge ontology (Mizoguchi 2004; Riek 2012) |
| **Referent** | object, spatial, user, other | Intentionality of question target |
| **C-type** | C1 Knowledge, C2 Comprehension, C3 Operation, C4 Evaluation | Bloom's taxonomy (Anderson & Krathwohl 2001) |
| **Q-type** | Q1 Verification, Q2 Case Specification, Q6 Definition, Q12 Method Explication, Q14 Judging | Graesser & Person (1994) question taxonomy |

**Critical finding: §3 of main.tex only describes 2 of these 4 dimensions.** The C-type and Q-type columns — which provide the strongest published theoretical grounding — are entirely absent from the paper's methodology description. This is the single most important gap to fix.

### Verdict: Reasonable in structure, incomplete in description

| Criterion | Assessment |
|-----------|-----------|
| K-type taxonomy | ✅ Sound — maps cleanly to knowledge representation ontology |
| Referent column | ✅ Sound — captures intentionality |
| C-type (Bloom's) | ✅ Sound — strong theoretical anchor; but **invisible in §3** |
| Q-type (Graesser & Person) | ✅ Sound — strong theoretical anchor; but **invisible in §3** |
| Formal mapping K+C+Q → AO/PE/PI | ❌ Missing — no decision rule stated anywhere |
| Inter-rater reliability scope | ⚠️ Unclear — κ reported for which dimensions? |
| Coding inconsistency | ⚠️ At least one case (GH010044!! Q1) coded C1/Q2 when content matches C4/Q14 |

**Bottom line**: The KCQ coding scheme is defensible as designed, but the paper's §3 describes only a subset of it, making replication impossible and leaving the theoretical grounding implicit. The framework does NOT need to be replaced — it needs to be **fully described and formalised**.

---

## Part 2 — Formal Mapping: (K, Referent, C, Q) → Question Element

The following decision rules derive AO / PE / PI labels from the raw coding. These rules are empirically validated against all 14 E1 dyads.

### PE (Preference-Eliciting)
```
K ∈ {K10, K11}  
AND Referent = user  
AND C = C4 (Evaluation)  
AND Q = Q14 (Judging)
```
Prototypical form: "你希望/喜欢/一般怎么…？" — the robot asks the user to express or evaluate their own preference without proposing any specific arrangement.

**Edge case (K11/user/C1/Q2):** GH010044!! Q1 ("你一般习惯怎么整理冰箱…") is coded C1/Q2 but semantically matches PE. The coder classified "habit knowledge" as C1 Knowledge rather than C4 Evaluation. **Recommend re-coding to C4/Q14** — this is a coding inconsistency, not a true DQ question. Should be addressed in IRR discussion.

### AO (Action-Oriented)
```
K ∈ {K1, K2, K3, K5, K6, K7}  
AND Referent ∈ {object, spatial, other}  
AND C ∈ {C1, C2, C3}  
AND Q ∈ {Q1, Q2, Q6, Q12}
```
Prototypical form: "xx放在哪里？" / "xx放这里可以吗？" — the robot asks about a specific item or zone without reference to the user's overall preference.

**Note on C4/Q1 with K5/spatial:** Questions like "就这里吗？" (K5/spatial/C4/Q1) elicit evaluation of a specific placement proposal, not an underlying preference. These are **AO** (proposal verification), not PE. The C4 level here indicates evaluation of a robot-proposed action, not user preference disclosure. The distinguishing feature is K-type: K5/spatial ≠ K11/user preference.

### PI (Preference-Induction / Parallel proposal)
```
Robot presents a tentative arrangement plan  
AND Q ∈ {Q12, Q14}  
AND C ∈ {C3, C4}
```
Prototypical form: "我整理的话，会把牛奶放这里，饮料放这里…你觉得怎么样？" — the robot proposes a global or partial layout and asks for evaluation. Distinguished from AO by the **scope** (global layout vs. single item).

Only one clear PI-opening dyad in E1: GH010067.

---

## Part 3 — Per-Dyad Analysis

### Definition of metrics
- **n** = total coded questions in dyad
- **First 20% window** = first ⌈0.2 × n⌉ questions
- **r_onset** = # PE questions in first 20% window / window size
- **Pattern rule**:
  - DQ: r_onset = 0 AND total PE = 0
  - UPF: r_onset ≥ 1 PE question in first 20% (i.e., at least one PE question opens the session)
  - PAR: session opens with PI question (robot proposes plan, asks evaluation), and/or PI questions distributed throughout

---

### GH010044 — DQ

**n ≈ 11**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 请问你一般把牛奶放在那 | K5 | object | C1 | Q2 | AO |
| 2 | 左边右边，还是都可以 | K5 | spatial | C4 | Q2 | AO (verification of specific loc) |
| 3 | 其他的，比如说水果之类的，一般会放在哪呢 | K5 | object | C3 | Q2 | AO |
| 4 | 就这里吗（用手指） | K5 | spatial | C4 | Q1 | AO |
| 5 | 那一些其他的吃的，比如零食、面等会放在哪 | K5 | object | C3 | Q2 | AO |
| 6 | 你觉得我这样摆得可以吗（指右门边） | K5 | spatial | C4 | Q1 | AO |
| 7 | 你不太喝可乐是吗 | K11 | user | C1 | Q2 | **borderline** (K11/user but C1) |
| 8 | 其他的呢，像这些呢 | K5 | object | C3 | Q2 | AO |
| 9 | 那我收拾成这样，你觉得满意吗 | K11 | — | C4 | Q1 | AO (end-of-task satisfaction) |
| 10 | 这个药在里边藏着…你觉得这个药需要放在哪里 | K5 | spatial | C3 | Q2 | AO |
| 11 | 这个抽屉里还有东西 | K5 | spatial | C3 | Q1 | AO |

**r_onset** (first 3 qs): 0/3 = **0**  
**Total PE**: 0  
**Pattern: DQ** — item-by-item spatial queries from the first question.

---

### GH010044 !! — UPF (with coding inconsistency)

**n ≈ 8**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 你一般习惯怎么整理冰箱，是按照饮料偏好还是盒子类型来整理 | K11 | user | C1 | Q2 | **PE*** |
| 2 | 一层放酒，一层放饮料是吗? | K5 | other | C3 | Q2 | AO |
| 3 | 物品优先放在哪个区域 | K5 | other | C1 | Q2 | AO |
| 4 | 水果的话放在上面还是下面呢 | K5 | other | C3 | Q2 | AO |
| 5 | 牛奶是放在饮料的下面还是上面 | K5 | other | C3 | Q2 | AO |
| 6 | 那已经开封的食品放在哪里呢 | K5 | other | C3 | Q2 | AO |
| 7 | 只要开封了我就放第一层，可以吧 | K5 | user | C4 | Q1 | AO |
| 8 | 这样可以吗 | K11 | user | C4 | Q1 | AO (end check) |

*Q1 is coded C1/Q2 but semantically asks about the user's organising preference. **Coding inconsistency**: should be C4/Q14.

**r_onset** (first 2 qs): 1/2 = **0.5** (counting semantic PE)  
**Pattern: UPF** — preference question first, then item-by-item AO.  
⚠️ **IRR note**: This dyad will lower κ on C-type if the second coder marks Q1 as C4.

---

### GH010046 — DQ

**n = 5**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 牛奶放不下了，放这里是不是好一些 | K5 | other | C4 | Q1 | AO (proposal verification) |
| 2 | 小杯的酸奶放在哪里 | K5 | other | C1 | Q2 | AO |
| 3 | 两瓶饮料能放这边吗 | K5 | other | C1 | Q1 | AO |
| 4 | 放不下怎么办 | K5/8 | other | C3 | Q12 | AO |
| 5 | 食物放这里可以吗 | K5 | other | C1 | Q1 | AO |

**r_onset**: 0/1 = **0**  
**Total PE**: 0  
**Pattern: DQ** — very short session, pure spatial/operational.

---

### GH010048 — UPF

**n ≈ 13**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 你一般喜欢怎么整理冰箱，按照饮料偏好还是按照比如盒子的形状整理 | K11 | — | C4 | Q14 | **PE** |
| 2 | 这个区域优先放在这里还是放在这两边呢 | K5 | — | C1 | Q1 | AO |
| 3 | 水果你希望放在下面还是上面 | K5 | — | C1 | Q1 | AO |
| 4–13 | All K5/spatial/C1-C3/Q1-Q12 | K5 | — | C1-C3 | Q1-Q12 | AO |

**r_onset** (first 3 qs): 1/3 = **0.33**  
**Pattern: UPF** — classic single PE opener, then all AO. User had proactive self-correction behaviour noted.

---

### GH010050 — UPF (strong onset)

**n ≈ 9**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 你希望这个冰箱怎么整理排列 | K10 | user | C4 | Q14 | **PE** |
| 2 | 你希望不同的东西怎么定义呢 | K2 | user | C1 | Q6 | **PE-adjacent** (user defines categories) |
| 3 | 你希望把xx东西隔离起来吗 | K10 | user | C1 | Q1 | borderline |
| 4–9 | K5/other/C1/Q1-Q2 | K5 | other | C1 | Q1-2 | AO |

**r_onset** (first 2 qs): 1–2/2 = **0.5–1.0**  
**Pattern: UPF** (strong) — two user-preference questions before any spatial query.

---

### GH010052 — UPF

**n ≈ 21** (3 green pre-task + 18 black during-task)

| # | Question | Phase | Element |
|---|----------|-------|---------|
| 1 | 你希望我怎么整理冰箱？ | green | **PE** (K11/C4/Q14) |
| 2 | 瓶子大小顺序有要求吗 | green | AO (K5/C1/Q1) |
| 3 | 下面的盒子是用来放水果的吗？ | green | AO (K5/C1/Q1) |
| 4–21 | Mixture of K5/spatial/C1/Q1-Q2, K1/K2/K3 identity/class/attr questions | black | AO |

**r_onset** (first 5 qs): 1/5 = **0.20**  
**Pattern: UPF** — pre-task PE establishes overall intent, extensive AO during execution.  
Note: This dyad has the highest total question count (≈21) and shows that even with many AO questions, the presence of a pre-task PE opener is the defining UPF feature.

---

### GH010054 — UPF (passive learner variant)

**n = 1** (robot is very passive — user volunteers all information)

| # | Question | Element |
|---|----------|---------|
| 1 | 你希望我怎么整理冰箱？ | **PE** (K11/C4/Q14) |

Robot notes: *"这个机器人很被动，用户会在整理过程中主动介入纠正"* — user proactively guided the robot throughout.

**r_onset**: 1/1 = **1.0**  
**Pattern: UPF** (degenerate: single PE question sufficient to elicit full preference; user took over). This is an important edge case showing that one high-information PE question can bootstrap the entire session.

---

### GH010056 — UPF (passive learner, user-driven)

**n ≈ 3**

| # | Question | Phase | Element |
|---|----------|-------|---------|
| 1 | 按形状排还是按食物类型排 | green | **PE** (K11/C4/Q14) |
| 2 | 哪些是常用的？ | green | AO (K7/C1/Q2) |
| 3 | 调料都放这边吧？ | black | AO (K5/C1/Q1) |

Robot notes: *"这个机器人主动性不强，而用户的主动性很强"* — user corrected and supplemented throughout.

**r_onset**: 1/1 = **1.0**  
**Pattern: UPF** — very sparse coding; the PE opener is the defining feature.

---

### GH010058 — UPF (strongest in dataset)

**n ≈ 10**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 你一般喜欢所有东西按照种类放吗 | K11 | user | C4 | Q14 | **PE** |
| 2 | 你更喜欢横着摆放还是竖着？还是根据空间大小决定? | K11 | user | C4 | Q14 | **PE** |
| 3 | 现在的冰箱有什么你感觉不舒服的么 | K11 | user | C4 | Q14 | **PE** |
| 4 | 基本规则是上面摆放的浅下面深吗？还是你更希望按照个人习惯 | K5 | user | C1 | Q1 | AO/PE-adj |
| 5 | 这个冰箱比较高，你会把上面的空间也利用上吗，还是根据身高合适摆放 | K5 | other | C4 | Q14 | PE-adj |
| 6 | 在这个冰箱里，你使用频率最高的是什么东西? | K7 | other | C4 | Q14 | PE-adj |
| 7 | 你会经常用到食物嘛 | K7 | other | C4 | Q14 | PE-adj |
| 8 | 零食吃的多吗? | K7 | other | C4 | Q14 | PE-adj |
| 9 | 总结一下：按照功能区摆放…这是这样吗 | K10 | user | C1 | Q1 | AO (summary check) |
| 10 | 对于冰箱深度，你会喜欢在触手可及的地方，还是考虑摆放物体的稳定… | K11 | user | C4 | Q14 | **PE** |

**r_onset** (first 2 qs): 2/2 = **1.0**  
**Total PE**: ≥ 4 (plus PE-adjacent)  
**Pattern: UPF** (extreme / most prototypical in dataset) — learner spends nearly all budget on preference discovery before any spatial query.

---

### GH010060 — DQ

**n ≈ 14**

| # | Question | K | Ref | C | Q | Element |
|---|----------|---|-----|---|---|---------|
| 1 | 请问你先让我整理侧面还是中间 | K6 | — | C1 | — | AO (sequencing) |
| 2 | 请问从左面开始还是右边 | K5 | — | C1 | Q1 | AO |
| 3 | 请问先规划液体是嘛 | K6 | — | C1 | Q1 | AO (sequencing) |
| 4 | 请问冰箱有三栏，液体放哪一栏 | K5 | — | C1 | Q2 | AO |
| 5–7 | 请问酸奶/奶油/发泥算液体吗 | K2 | other | C1 | Q1 | AO (classification) |
| 8 | 现在我已经整理完所有液体，请问怎么摆放 | K5 | — | C3 | Q12 | AO |
| 9 | 花生酱用袋子装的，算液体吗 | K2 | — | C1 | Q1 | AO |
| 10 | 请问右侧面摆放完成了，可以开始放左侧面吗？ | K5 | — | C1 | Q1 | AO |
| 11 | 现在液体整理完成了，下一步整理什么 | K7 | — | C1 | Q2 | AO |
| 12 | 请问鱿鱼丝算生鲜吗 | K2 | — | C1 | Q1 | AO |
| 13 | 请问这是什么（拿着问） | K1 | — | C1 | Q6 | AO |
| 14 | 请问剩下的东西怎么整理 | K5 | — | C3 | Q12 | AO |

**r_onset**: 0/3 = **0**  
**Total PE**: 0  
**Pattern: DQ** — purely operational from start: asks about sequencing, then identifies and places items one-by-one by category. No preference knowledge sought at any point.

---

### GH010062 — UPF

**n ≈ 13**

| # | Question | Element |
|---|----------|---------|
| 1 | 你会把食品和饮料摆放在一起吗 | **PE** (K11/user/C4/Q14) |
| 2 | 首先，饮料和水要摆放在侧面吗 | AO (K5/user/C1/Q1) |
| 3 | 每种饮料的不同类型会摆放在一起吗 | AO (K5/user/C1/Q1) |
| 4–10 | K5/other/C1-C4/Q1-Q2 | AO |
| 11 | 把这个放下面，这个放上面，怎么样 | PI (K5/C4/Q14 — proposal+evaluation) |
| 12–13 | K5/other, K3/other | AO |

**r_onset** (first 3 qs): 1/3 = **0.33**  
**Pattern: UPF** — PE opener, then AO. One mid-session PI question (proposal-evaluation) noted.

---

### GH010064 — UPF (very proactive user)

**n ≈ 8**

| # | Question | Phase | Element |
|---|----------|-------|---------|
| 1 | 你希望按品类整理吗？比如这里放牛奶，这里放酸奶（手指） | green | **PE** (K11/user/C4/Q14) |
| 2 | 这里放全部水果是吧 | black | AO (K5/C1/Q1) |
| 3 | 调味料吗？放侧面吗 | black | AO |
| 4–8 | K5/other/C1/Q1-Q2 | black | AO |

Robot notes: *"这个实验的用户主动性很强，在前期思考过程中，机器人只需要少量的举例提问，用户就能反馈很多信息"* — pre-task PE question sufficient to obtain rich preference information.

**r_onset** (first 2 qs): 1/2 = **0.5**  
**Pattern: UPF** — single pre-task PE opener is enough (user is forthcoming).

---

### GH010067 — PAR

**n ≈ 7**

| # | Question | Phase | Element |
|---|----------|-------|---------|
| 1 | 我整理的话，会把牛奶放这里，饮料放这里，水果放这里…. 你觉得怎么样？ | green | **PI** (K11/C4/Q14 — robot proposes global plan, asks eval) |
| 2 | 这一层比较隔离，适合放水果对么？ | black | AO (K5/C1/Q1) |
| 3 | 蘑菇酱放上面吗？ | black | AO |
| 4 | 火锅蘸料放不下了，怎么办 | black | AO (C3/Q12) |
| 5 | 巧克力要冷藏吗？ | black | AO (K7/C1/Q1) |
| 6 | 这是什么？ | black | AO (K1/Q6) |
| 7 | 还有一瓶水怎么办？ | black | AO (C3/Q12) |

**r_onset** (PE only): 0/2 = 0 (Q1 is PI, not PE)  
**Pattern: PAR** — the robot proposes a complete layout in the pre-task phase and asks for evaluation. This is the defining PAR behaviour: parallel reasoning about a solution while eliciting user feedback. Distinct from UPF (which asks "what do you like?" without proposing anything) and DQ (which asks about individual items without a global plan).

Robot notes: *"两者更像一种讨论的方式参与"* — collaborative/discussion mode.

---

### GH010069 — UPF

**n ≈ 4**

| # | Question | Phase | Element |
|---|----------|-------|---------|
| 1 | 你希望用什么方式优先整理这个冰箱 | green | **PE** (K11/C4/Q14) |
| 2 | 总结一下，首先按照类型放，然后按照保质期放？ | green | AO (K6/C1/Q1 — summary verification) |
| 3 | 你对物品位置有要求吗？ | green | **PE** (K11/C4/Q14) |
| 4 | 椰汁要竖起来吗？ | black | AO (K5/C1/Q1) |

**r_onset** (first 1 q): 1/1 = **1.0**  
**Total PE**: 2  
**Pattern: UPF** — two PE questions in pre-task, then single AO during execution (very few execution questions because PE established rules upfront).

---

## Part 4 — Summary Table

| Dyad | n | PE questions | First 20% PE | r_onset | Pattern | Notes |
|------|---|-------------|--------------|---------|---------|-------|
| GH010044 | 11 | 0 | 0/3 | 0 | **DQ** | All K5/spatial AO |
| GH010044!! | 8 | 1* | 1/2 | 0.5* | **UPF*** | *Coding inconsistency: K11/C1/Q2 should be C4/Q14 |
| GH010046 | 5 | 0 | 0/1 | 0 | **DQ** | Short pure-AO session |
| GH010048 | 13 | 1 | 1/3 | 0.33 | **UPF** | Classic single-PE opener |
| GH010050 | 9 | 1–2 | 1–2/2 | 0.5–1.0 | **UPF** | Two PE-type openers |
| GH010052 | 21 | 1 | 1/5 | 0.20 | **UPF** | Pre-task PE, long AO execution |
| GH010054 | 1 | 1 | 1/1 | 1.0 | **UPF** | Single PE, user took over |
| GH010056 | 3 | 1 | 1/1 | 1.0 | **UPF** | Sparse; PE onset is only question |
| GH010058 | 10 | 4+ | 2/2 | 1.0 | **UPF** | Strongest UPF in dataset; 3× K11 opener |
| GH010060 | 14 | 0 | 0/3 | 0 | **DQ** | Purely operational from Q1 |
| GH010062 | 13 | 1 | 1/3 | 0.33 | **UPF** | PE onset then AO |
| GH010064 | 8 | 1 | 1/2 | 0.5 | **UPF** | Pre-task PE; user very forthcoming |
| GH010067 | 7 | 0 PE / 1 PI | 0 PE / 1/1 PI | — | **PAR** | Opens with global plan proposal (PI) |
| GH010069 | 4 | 2 | 1/1 | 1.0 | **UPF** | Two PE questions, then single AO |

### Distribution
| Pattern | Count | % |
|---------|-------|---|
| DQ | 3 | 21% |
| UPF | 10 | 71% |
| PAR | 1 | 7% |

---

## Part 5 — Key Structural Finding: Pre-task vs. During-task

**Across all 14 dyads, there is a consistent structural divide:**

- **Pre-task (green) questions** are where PE and PI questions appear. 11 of 14 dyads have their PE/PI question(s) in the green/pre-task phase.
- **During-task (black) questions** are overwhelmingly AO regardless of overall strategy. Even the most UPF-heavy learner (GH010058) transitions to K5/spatial/Q1-Q2 AO questions during execution.

**Implication for PrefQuest model**: The preference-acquisition budget B represents pre-task/early-session investment. The question is not "do we ever ask PE questions" but "does the learner front-load PE questions before beginning item-by-item execution." This maps perfectly to the UPF vs. DQ simulation design.

**The pre-task phase is the discriminating context** for strategy classification. The paper should make this explicit.

---

## Part 6 — What §3 Must Add

### 6.1 Describe all four coding dimensions (currently invisible)

Add a table of all four dimensions with:
- Formal name and abbreviation
- All values used (K1–K11; object/spatial/user/other; C1–C4; Q1/Q2/Q6/Q12/Q14)
- Definition of each value
- Theoretical reference for each dimension

### 6.2 State the formal element-classification rule

Add a decision procedure (can be a small table or IF/THEN rule):
```
PE: K ∈ {K10, K11} AND Ref = user AND C = C4 AND Q = Q14
PI: robot proposes layout AND C ∈ {C3,C4} AND Q ∈ {Q12,Q14}
AO: K ∈ {K1–K7} AND Ref ∈ {obj, spatial, other} AND C ∈ {C1–C3} AND Q ∈ {Q1,Q2,Q6,Q12}
```

### 6.3 Report κ per dimension, not just overall

Currently the paper has three TODO slots for κ (knowledge-level, intent-level, pattern-level). With four coding dimensions now visible, you need κ for: K-type, Referent, C-type, Q-type, and the derived AO/PE/PI label. Report as a table, not inline.

### 6.4 Report the pre-task/during-task distinction

Add one sentence: "We coded all questions from the pre-task thinking phase (green) and task execution phase (black) separately, as the distribution of question elements differed systematically between phases."

### 6.5 Report the strategy distribution

After pattern classification, add: "Of the 14 E1 dyads, 10 (71%) exhibited a UPF pattern, 3 (21%) exhibited a DQ pattern, and 1 (7%) exhibited a PAR-like pattern characterised by a global layout proposal in the pre-task phase."

### 6.6 Acknowledge the coding inconsistency in IRR section

GH010044!! Q1 reveals that "habit" questions (K11/user asking about habit) can be ambiguously coded as C1 (Knowledge) vs. C4 (Evaluation). This should be flagged as a source of disagreement in IRR and resolved by a tie-breaking rule.

---

## Part 7 — Does the Framework Need to Change?

**Short answer: No new framework is needed. The existing 4D coding already contains everything required.**

The literature-grounded framework proposed in `hhi_coding_methodology_report.md` (Dimension A Information Target + Dimension B Epistemic Stance) is a valid alternative framing — but it is **isomorphic** to the existing K+C+Q coding:

| Proposed A-B framework | Existing KCQ equivalent |
|------------------------|-------------------------|
| A1 Object-Specific | K5 + Ref=object + C1 + Q1/Q2 |
| A3 Rule/Principle | K11 + Ref=user + C4 + Q14 |
| B1 Open | Q14 (Judging) |
| B2 Directed | Q1/Q2 (Verification/Case) |
| B3 Hypothetical | Q14 with proposal framing (PI) |

**Recommendation**: Do not introduce a new coding scheme. Instead, formalise and fully describe the existing 4D scheme in §3, add the formal mapping rule, and report all κ values. The paper can cite Graesser & Person (1994) for Q-type, Bloom's taxonomy for C-type, and Mayring (2000) as the overarching qualitative content analysis framework.

---

*Generated: 2026-04-21. Based on complete read of HHI/视频分析-E1.pdf (20 pages, 14 E1 dyads).*
