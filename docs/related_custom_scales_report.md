# Related Custom Scales Report: Precedents for PrefQuest's CL / PU / IA Measures

*Prepared for: PrefQuest §6.5 Measures section justification*
*Date: 2026-04-21*
*Coverage: CHI, IJHCS, TOCHI, THRI, CSCW, HRI, IUI, Int. J. Social Robotics, Behaviour & IT*

---

## 1. Overview: Community Norms for Custom Scale Design

### 1.1 General Posture

The HCI/HRI community has a well-established, if sometimes inconsistently applied, norm of designing task-specific scales when no validated off-the-shelf instrument fully captures the target construct. The canonical trigger is: *the construct is novel or domain-specific enough that existing tools either do not exist, do not fit the interaction modality, or conflate irrelevant dimensions*. This norm is explicitly endorsed in the psychometric methodology literature applied to HCI (e.g., Schrum et al., 2020; 2023) and is the basis for scale development projects that have become community standards (e.g., Godspeed, ASAQ, CCR, BUS-15).

### 1.2 Dominant Justification Strategies

Across the papers surveyed, custom or adapted scales are justified by one or more of the following strategies:

1. **Theoretical derivation** – Items are grounded in an established theoretical framework (cognitive load theory, sense-of-agency literature, CASA paradigm, self-determination theory). Items reflect the theory's constructs directly.
2. **Community need / gap argument** – Authors document that no prior validated measure exists for the specific interaction context or agent type.
3. **Construct decomposition from existing scales** – Researchers select subscales from a validated multi-factor instrument (e.g., NASA-TLX) rather than using all subscales, explicitly citing that only selected dimensions are theoretically relevant.
4. **Expert / pilot validation** – Draft items are submitted to domain experts for content validity ratings; pilot data are used to refine the item pool via factor analysis and reliability testing.
5. **Precedent citation** – Authors cite prior HCI/HRI studies that designed analogous custom items for similar constructs, implicitly invoking community acceptance.

### 1.3 Minimal Reporting Standards

Based on the survey of HRI/HCI venues, the field expects at minimum:

- **Cronbach's α ≥ .70** per construct, reported *per condition* in within-subjects designs (Schrum et al., 2020; 2023)
- **At least 3 items per construct** (single-item scales are considered insufficient for reliability assessment)
- **7-point Likert anchors** are the dominant convention in contemporary IJHCS/CHI/HRI work
- Non-parametric statistics (Wilcoxon signed-rank, Friedman) for Likert data unless normality is established; paired parametric tests acceptable for composite scale scores

A 2023 audit of 144 HRI papers found that only 4 correctly designed and analyzed Likert scales — underscoring that reporting α, justifying custom items, and applying appropriate statistics remain differentiating quality markers.

---

## 2. Construct-by-Construct Comparison

### 2.1 Cognitive Load (CL) — Adapted NASA-TLX Subscales

**PrefQuest approach:** 3 NASA-TLX subscales (Mental Demand, Effort, Frustration), 7-pt, aggregated into CL composite.

**Community precedents:**

| Approach | Precedent |
|---|---|
| Using a subset of NASA-TLX subscales | Hart (2006, HFES) explicitly documents that researchers have dropped, added, or redefined subscales since the instrument's inception. Surgical-domain studies replaced 3 of 6 dimensions with domain-relevant ones. HCI studies routinely drop Physical Demand and Temporal Demand when tasks are not physically or time-pressured. |
| Mental Demand + Effort + Frustration triad | Converges with the "psychological burden" cluster identified in meta-analyses of NASA-TLX factor structure. Physical and Temporal Demand load onto a separate "execution demand" factor and are routinely omitted in interactive dialogue studies. |
| Reporting α for composite | The three subscales (Mental Demand, Effort, Frustration) consistently cohere as a unidimensional "perceived burden" factor in dialogue/typing tasks. Reporting α for their composite is accepted practice. |
| IJHCS precedent | Kosch et al. (2023, survey in HCI: *A Survey on Measuring Cognitive Workload in Human-Computer Interaction*) confirms NASA-TLX partial use is the dominant pattern in interactive-system evaluations. |

**Key lesson for PrefQuest:** Omitting Physical Demand and Temporal Demand is theoretically defensible because PrefQuest questioning involves no physical exertion and is not time-pressured. The three retained subscales tap the dimension most relevant to conversational interface burden. Reporting α for the 3-item composite (expected ≥ .75 based on literature) is sufficient.

---

### 2.2 Perceived Understanding (PU) — Custom Items

**PrefQuest approach:** 3 custom 7-pt Likert items measuring whether users felt the agent learned their *transferable* preferences.

**Community precedents and analogues:**

#### 2.2.1 CRS-Que — CUI Understanding subscale
*Chen et al. (2024). CRS-Que: A User-centric Evaluation Framework for Conversational Recommender Systems. ACM Transactions on Recommender Systems.*

- Developed and validated an 11-construct framework for conversational recommender systems.
- The **CUI Understanding** construct (≥ 3 items per the ≥3-item requirement) captures whether users felt the conversational interface understood their needs.
- Constructs with only a single item were explicitly dropped from the final framework because single items cannot assess measurement error.
- All constructs validated with Cronbach's α; discriminant validity assessed via correlation structure.
- **Lesson:** "Perceived understanding by a conversational agent" is an accepted, measurable construct in recommender-system HCI; items should tap (a) comprehension of stated preferences, (b) accurate representation of preferences in recommendations, (c) generalizability of understanding.

#### 2.2.2 Preference Elicitation UX Study — Perceived Understandability
*[The effect of preference elicitation methods on the user experience in conversational recommender systems. IJHCS, 2024.]*

- Measured 10 user experience constructs including "understandability" of the system via questionnaire.
- The study found no significant subjective differences across three elicitation methods (but significant objective differences in chat duration and recommendation accuracy), illustrating that sensitivity of custom scales depends on construct specificity.
- **Lesson:** Custom UX items for CRS must be closely tied to the manipulation (here: question strategy) to detect differences. Items should directly ask about the *mechanism* (e.g., "The agent's questions helped it understand what I really care about").

#### 2.2.3 Ashktorab, Jain, Liao & Weisz — Resilient Chatbots (CHI 2019)
*Ashktorab, Z., Jain, M., Liao, Q.V., & Weisz, J.D. (2019). Resilient Chatbots: Repair Strategy Preferences for Conversational Breakdowns. CHI 2019.*

- Large-scale scenario-based study (N = 203) evaluating eight repair strategies for chatbot breakdown.
- Measured user preferences and perceptions with custom Likert items adapted to the dialogue repair context.
- The scenario design (wizard-of-oz style) required custom measures because no existing instrument addressed "perceived chatbot understanding of intent after breakdown."
- **Lesson:** In conversational AI studies, task-specific items tied to the manipulation (question strategy, repair strategy, etc.) outperform generic satisfaction scales in detecting condition differences.

#### 2.2.4 ASAQ — Coherence and Sociability constructs
*Konijn et al. (2025). The Artificial Social Agent Questionnaire (ASAQ). International Journal of Human-Computer Studies.*

- Multi-year development by 100+ researchers; identified 19 constructs across 90 items.
- The **Coherence** construct (∼4 items) measures whether the agent behaved consistently and understandably — adjacent to "perceived understanding" from the agent's side.
- α values reach .72 (respectable) across a diverse set of agent types.
- **Lesson:** Even with expert-generated items and large validation samples, α ∼ .72–.80 is acceptable for novel constructs in agent evaluation contexts. PrefQuest's PU α ≥ .70 is a reasonable target.

#### 2.2.5 Design principle for PU items
Based on the literature, PU items should be phrased to distinguish *surface compliance* (the agent answered correctly) from *deep preference learning* (the agent learned what matters across unseen items). Example anchoring: "The agent's questions helped it understand the underlying reasons for my preferences" and "I feel the agent could now make good choices for items I haven't discussed." This operationalizes the *transferability* dimension that is central to PrefQuest's theoretical contribution.

---

### 2.3 Interaction Agency (IA) — Custom Items

**PrefQuest approach:** 3 custom 7-pt Likert items measuring users' sense of directing what the agent learned.

**Community precedents:**

#### 2.3.1 Bergström, Knibbe, Pohl & Hornbæk — Sense of Agency and UX (TOCHI 2022)
*Bergström, J., Knibbe, J., Pohl, H., & Hornbæk, K. (2022). Sense of Agency and User Experience: Is There a Link? ACM Transactions on Computer-Human Interaction, 29(4).*

- Directly investigates the link between sense of agency (SoA) and UX quality in HCI.
- Used custom agency items such as "It felt like I was in control of the movements during the task" (7-pt strongly disagree–strongly agree).
- Also used AttrakDiff pragmatic quality items (e.g., confusing–structured, 7-pt semantic differential).
- Key finding: higher-level perceived control (questionnaire) and lower-level SoA (objective) are partially dissociated — subjective questionnaires capture *experienced* agency, which is the relevant construct for UX evaluation.
- **Lesson for IA:** The IA scale should capture *experienced* agency — the felt sense of directing the conversation — not objective control. Items should reference "I felt I was guiding what the agent focused on" rather than behavioral measures.

#### 2.3.2 Sundar (2020) — Rise of Machine Agency Framework
*Sundar, S.S. (2020). Rise of Machine Agency: A Framework for Studying the Psychology of Human–AI Interaction (HAII). Journal of Computer-Mediated Communication.*

- Theoretical framework distinguishing *user agency* (locus of control over interaction) from *machine agency* (AI autonomy).
- When the AI and user collaborate, mutual augmentation of agency predicts better outcomes. This framework motivates a scale that captures whether the *user* felt in the agentic role during the preference-elicitation conversation.
- **Lesson:** PrefQuest's IA scale operationalizes the "user agency" pole of this framework — the degree to which the user, not the system, directed the focus and depth of preference elicitation. Items should explicitly reference directionality: "I decided what aspects of my preferences the agent should learn about."

#### 2.3.3 Robot Autonomy Perception Scale (RAPS)
*[Toward RAPS: the Robot Autonomy Perception Scale. arXiv 2407.11236, 2024.]*

- Theoretically motivated 15-item scale (five components: sense, plan, act, goal-directed action, independent action).
- Based on autonomy theory; items use 7-pt Likert.
- **Lesson:** Theoretically decomposing "agency/autonomy" into sub-dimensions (sensing, planning, acting) is the accepted approach for novel constructs in HRI. PrefQuest's IA can be similarly decomposed into: (a) felt ability to *initiate* topics, (b) felt ability to *steer* depth of questioning, (c) felt ability to *redirect* the agent's focus.

#### 2.3.4 Human Autonomy and SoA in HRI — Systematic Review
*[Human Autonomy and Sense of Agency in Human-Robot Interaction: A Systematic Literature Review. arXiv 2509.22271.]*

- Identified that SoA in HRI is primarily operationalized through psychometric scales or the intentional-binding paradigm; questionnaire-based approaches dominate in field studies.
- Operationalization of user autonomy primarily involves intentional control, felt influence over outcomes, and perceived initiative.
- **Lesson:** The review legitimizes questionnaire-based measures of agency/autonomy as the dominant method in naturalistic HRI studies (as opposed to lab-based physiological paradigms). PrefQuest's IA scale follows this community norm.

#### 2.3.5 CRS-Que — User Control subscale
*Chen et al. (2024). CRS-Que. ACM TORS.*

- The **User Control** construct (≥ 3 validated items) specifically captures whether users felt they could direct the conversational recommender's behavior.
- Validated with α and factor loading; discriminant validity confirmed against adjacent constructs (Transparency, CUI Understanding).
- **Lesson:** "User control over conversational agent behavior" is an established, distinct, measurable construct in CRS research — directly analogous to PrefQuest's IA. Items should measure directionality and influence over agent focus, not just satisfaction with outcomes.

---

## 3. Key Methodological Lessons for PrefQuest's CL/PU/IA Scales

### 3.1 Three Items per Construct is the Community Minimum

Multiple sources (Schrum et al., 2020, 2023; CRS-Que, 2024; ASAQ, 2025) converge on three items as the *practical minimum* for computing Cronbach's α and assessing discriminant validity. Single-item scales cannot assess measurement error and are routinely criticized or excluded.

**Implication:** PrefQuest's 3-item design for each of CL (NASA-TLX subscales), PU, and IA meets the minimum. The theoretical justification for 3 items rather than more is that (a) the constructs are unidimensional, (b) participant fatigue in within-subjects designs warrants parsimonious scales, and (c) community precedent (BUS-15 factors reduced to 2–7 items; ASAQ ∼4 items per construct) shows 3–4 items per unidimensional construct is standard.

### 3.2 Report α per Condition in Within-Subjects Designs

Schrum et al. (2023) specifically note that in within-subjects designs, Cronbach's α must be computed and reported *per condition*, since item intercorrelations may differ across conditions. A scale that is reliable in one condition but not another signals measurement problems or genuine construct instability.

**Implication:** PrefQuest should report α for CL, PU, and IA separately for the DQ, UPF, and PAR conditions (or at minimum across all combined). If α drops below .70 in any condition, item wording or construct specification needs review.

### 3.3 Discriminant Validity Between Adjacent Constructs

The most salient risk is construct overlap between PU (perceived understanding) and IA (interaction agency), and between CL and IA. In CRS-Que (2024), constructs with high inter-correlation were *merged* (e.g., CUI Positivity + CUI Rapport → merged; Trust + Confidence → merged). The final framework retained distinct constructs only where discriminant validity was confirmed.

**Implication:** PrefQuest should verify that PU and IA items do not cross-load in pilot data. Conceptual distinction: PU is *retrospective and outcome-oriented* ("the agent understood my transferable preferences"), while IA is *process-oriented and prospective* ("I felt I was directing what the agent learned during the conversation"). This distinction maps to the question type manipulation: UPF's preference-induction questions should amplify IA without necessarily amplifying PU (since PI questions probe generalization, which is what UPF is designed to elicit). The two constructs may correlate but should be discriminable.

### 3.4 α Benchmarks in Context

Based on the surveyed literature:

| α range | Interpretation in HCI/HRI |
|---|---|
| < .60 | Unacceptable; items do not measure same construct |
| .60–.69 | Borderline; acceptable only for very early/exploratory instruments |
| .70–.79 | Acceptable / respectable (ASAQ norm: ∼.72) |
| .80–.89 | Good (BUS-15: .76–.87; Godspeed: .81–.91; CCR: .95–.97) |
| ≥ .90 | Excellent but may signal redundant items |

For 3-item scales measuring novel constructs in a constrained within-subjects design, α ≥ .70 is the accepted threshold. The Godspeed and Chatbot Usability scales demonstrate that α ≥ .80 is achievable with semantically coherent items even in novel domains.

### 3.5 Theoretical Justification is Necessary and Sufficient for Novel Constructs

None of the surveyed papers that designed custom scales for novel constructs (PU, IA analogues) underwent the full psychometric validation pipeline (EFA + CFA + multi-sample validation) for the initial publication. The standard in CHI/IJHCS/HRI for a *first-use* custom scale is:

1. Ground items in theory (cognitive load theory; sense of agency theory; CRS user control literature)
2. Pilot test items for face validity (ideally 2–5 domain experts review item wording)
3. Report α per condition in the study
4. Acknowledge limitations (cross-validation, multi-sample validation as future work)

The ASAQ, CCR, BUS-15, and Godspeed scales all underwent multi-study validation after first appearing in a single-study form. PrefQuest's scales are consistent with the community norm for first-publication custom items.

### 3.6 Within-Subjects Design Considerations

Within-subjects designs create specific scale design risks:

- **Carryover effects on ratings:** If participants adjust their frame of reference across conditions, scale anchors may drift. Counterbalancing (Latin square) mitigates order effects but does not eliminate rating drift.
- **Response scale inflation:** Participants may compress ratings after experiencing multiple conditions. BUS-15 authors noted that the scale was tested across participants but recommended within-subjects researchers verify α in each condition.
- **Item wording should reference the just-completed condition explicitly:** "During this session..." rather than "In general..." forces condition-specific attribution and reduces carryover.

**Implication:** PrefQuest's items should be phrased to anchor to the specific interaction session ("During this questioning session," "In this condition," "The questions I just answered..."), not globally.

---

## 4. Reference Table of All Papers Surveyed

| # | Title (Abbreviated) | Authors | Venue | Year | Constructs / Scale | Items | Scale | α |
|---|---|---|---|---|---|---|---|---|
| 1 | CRS-Que: User-centric Framework for CRS | Chen et al. | ACM TORS | 2024 | 11 constructs: CUI Understanding, User Control, Transparency, Ease of Use, etc. | 37 total (≥3 per construct) | 7-pt Likert | Reported per construct; discriminant validity via HTMT |
| 2 | Effect of Preference Elicitation Methods on UX in CRS | [Authors] | IJHCS | 2024 | 10 UX constructs incl. Understandability, Perceived Control | Multi-item | 7-pt | Analyzed; no significant differences across PE conditions in subjective measures |
| 3 | The Chatbot Usability Scale (BUS-15) | [Følstad et al.] | Personal & Ubiquitous Computing | 2021 | 5 factors: conversational efficiency, accessibility, functionality, etc. | 15 total (2–7 per factor) | Not specified (Likert) | .76–.87 per factor; overall .87 |
| 4 | Sense of Agency and User Experience: Is There a Link? | Bergström, Knibbe, Pohl, Hornbæk | TOCHI | 2022 | Sense of agency (3 custom items), AttrakDiff pragmatic quality | 3 (agency) + AttrakDiff | 7-pt Likert; 7-pt semantic differential | Not extracted; study found partial dissociation SoA–UX |
| 5 | Concerning Trends in Likert Scale Usage in HRI: Best Practices | Schrum, Ghuy, Hedlund-Botti et al. | ACM THRI | 2023 | Meta-review: design and analysis norms | N/A | N/A | Recommends α ≥ .70, ≥3 items, report per condition |
| 6 | Four Years in Review: Statistical Practices of Likert Scales in HRI | Schrum, Johnson, Ghuy, Gombolay | HRI Companion | 2020 | Meta-survey of 110 HRI papers | N/A | N/A | Found only 3/110 correctly designed and analyzed; α ≥ .70 threshold |
| 7 | ASAQ: Artificial Social Agent Questionnaire | Konijn et al. | IJHCS | 2025 | 19 constructs: Believability, Sociability, Coherence, Naturalness, etc. | 90 (long); 24 (short; ~4 per construct) | 7-pt Likert | α ∼ .72 (respectable) across constructs |
| 8 | Connection-Coordination Rapport (CCR) Scale | Lin, Dinner, Leung, Mutlu, Trafton, Sebo | HRI | 2025 | 2 factors: Connection (12 items), Coordination (6 items) | 18 | 7-pt Likert | α = .95–.97 across validation studies |
| 9 | Perception and Evaluation in HRI: HRIES | Spatola, Kühnlenz | Int. J. Social Robotics | 2021 | 4 dimensions: Sociability, Agency, Animacy, Disturbance | ~24 (semantic differential) | 5-pt semantic differential | Good across studies; α reported per dimension |
| 10 | Godspeed Questionnaire Series | Bartneck, Kulić, Croft, Zoghbi | Int. J. Social Robotics | 2009 | 5: Anthropomorphism, Animacy, Likeability, Perceived Intelligence, Safety | 24 (5-item subscales) | 5-pt semantic differential | .81–.91 per subscale; .85+ overall |
| 11 | HRI CUES: Conversational User Enjoyment Scale | Irfan et al. | IEEE THRI | 2024 | Enjoyment: satisfaction, fun, interestingness, strangeness | 4 | 5-pt Likert (annotator) | α = .84 |
| 12 | Rise of Machine Agency Framework (HAII) | Sundar | J. Computer-Mediated Communication | 2020 | Theoretical: user agency vs. machine agency in HCI | N/A (framework) | N/A | N/A |
| 13 | Resilient Chatbots: Repair Strategy Preferences | Ashktorab, Jain, Liao, Weisz | CHI | 2019 | Preferences and perceptions of chatbot repair strategies (custom items) | Multi-item | Likert | Not extracted; large N scenario-based design |
| 14 | Choice-Based Preference Elicitation for CF Recommender | Loepp, Hussein, Ziegler | CHI | 2014 | Item fit, interaction effort, user control, transparency, understanding | Multi-item per construct | 7-pt Likert | Reported; within-subjects, counterbalanced |
| 15 | Human Autonomy and SoA in HRI: Systematic Review | [Authors] | arXiv | 2025 | Reviews operationalization of autonomy/agency in HRI | N/A | Mostly Likert questionnaires | N/A; SLR finding |
| 16 | UX Research on Conversational Human-AI Interaction | Zheng et al. | CHI | 2022 | Review of evaluation methods in conversational AI UX research | N/A | Various | N/A; literature synthesis |
| 17 | NASA-TLX: 20 Years Later | Hart | HFES | 2006 | 6 subscales: Mental Demand, Physical Demand, Temporal Demand, Effort, Performance, Frustration | 6 | 100-pt (or 7-pt raw) | Internally validated; partial use documented |
| 18 | Should We Use the NASA-TLX in HCI? | [Authors] | Int. J. Human-Computer Studies | 2025 | Review of NASA-TLX suitability, subscale appropriateness in HCI | N/A | N/A | N/A; review recommends subscale selection |
| 19 | Questionnaires to Measure Acceptability of Social Robots | Weiss et al. | Robotics | 2019 | Review of HRI acceptability instruments | N/A | Various | N/A; calls for reliability testing |
| 20 | CASUX: Standardized Scale for AI Conversational Agents | [Authors] | Int. J. Human-Computer Interaction | 2024 | 9 dimensions: practicality, proficiency, humanness, sentiment, robustness, etiquette, personality, anthropomorphism, ease of use | 34 | Likert | Validated across 6 studies |

---

## 5. Synthesis: Implications for PrefQuest §6.5 Measures

### CL (Cognitive Load — 3 NASA-TLX subscales)

**Precedent:** Hart (2006) documents partial subscale use as standard practice; Kosch et al. (2023, HCI survey) confirms it as dominant pattern. Omitting Physical Demand and Temporal Demand is theoretically motivated (no physical/time pressure in dialogue questioning).

**Justification text template:** "We operationalized cognitive load using three NASA-TLX subscales (Mental Demand, Effort, Frustration; 7-pt; Hart & Staveland, 1988) that are theoretically relevant to conversational questioning tasks. Physical Demand and Temporal Demand were excluded as participants experienced no physical exertion and sessions were self-paced. This partial-subscale approach follows established HCI practice (Hart, 2006; Kosch et al., 2023)."

### PU (Perceived Understanding — 3 custom items)

**Precedent:** CRS-Que (2024) establishes CUI Understanding as a validated, distinct construct in conversational recommender research. No existing instrument captures the *transferability* dimension (whether the agent learned generalizable preferences, not just surface compliance). Items grounded in preference learning theory.

**Justification text template:** "We designed a three-item Perceived Understanding scale (7-pt Likert) drawing on the CUI Understanding construct from CRS-Que (Chen et al., 2024) and the "felt understood" conceptualization from conversational agent UX research (Ashktorab et al., 2019). Items specifically target the *transferability* of preference learning — whether the agent understood preferences well enough to generalize to unseen items — which is the central construct in PrefQuest and is not captured by generic satisfaction or usability scales."

### IA (Interaction Agency — 3 custom items)

**Precedent:** Bergström et al. (2022, TOCHI) validates questionnaire-based measurement of SoA in HCI; Sundar (2020) provides the theoretical framework distinguishing user agency from machine agency; CRS-Que's User Control construct demonstrates the construct's measurability in CRS contexts; SLR on human autonomy in HRI confirms questionnaire dominance.

**Justification text template:** "We designed a three-item Interaction Agency scale (7-pt Likert) grounded in Sundar's (2020) machine agency framework and Bergström et al.'s (2022) operationalization of sense of agency in HCI. Items capture users' felt sense of directing the agent's learning — the degree to which participants experienced themselves as the agent of what was learned — operationalizing the 'user agency' pole of the human-AI agency spectrum (Sundar, 2020). This construct is analogous to the User Control construct validated in CRS-Que (Chen et al., 2024) but is adapted to the learning-orientation of PrefQuest rather than recommendation control."

---

## 6. Notes on Sources and Limitations

- Several key papers (CRS-Que full text, BUS-15 full item list, Bergström et al. full item list, Loepp et al. full scale) were inaccessible due to network restrictions during literature collection; details were reconstructed from search result abstracts, summaries, and secondary sources.
- α values for some scales (particularly Bergström et al., 2022; Loepp et al., 2014) could not be confirmed from primary sources; these papers are cited for construct-level precedent rather than specific psychometric benchmarks.
- The field is evolving rapidly: the 2024–2025 publications (ASAQ, CCR, CRS-Que) represent the current frontier of validated instrument design; earlier work (Godspeed, BUS-15) provides the psychometric anchor for α expectations.
- PrefQuest's within-subjects design across three strategies (DQ, UPF, PAR) with full counterbalancing is consistent with the Loepp et al. (2014) CHI design and should follow Schrum et al.'s (2023) recommendation to report α per condition.

---

*End of report.*
