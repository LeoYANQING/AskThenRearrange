# Study 2 Design Discussion — Reconciling Subjective Experience with Objective PSR

**Status:** Discussion draft, not yet merged into `study2_experiment_guide.md`.
**Purpose:** Think through how Study 2 should be designed so that user experience and objective generalization (unseen PSR) produce a *coherent* paper story instead of a mixed-signal result.

---

## 1. The Problem We're Trying to Avoid

Simulation (Study 1) already tells us UPF gives the best unseen PSR. The classical HCI failure mode is:

- Study 2 finds users *subjectively prefer* PAR (exploratory dialogue feels natural).
- Study 2 finds UPF still *objectively generalizes* best.
- Paper reports both and concludes "tradeoff between user experience and generalization."

That is a weak ending. Reviewers will ask: *so which should a designer actually pick?* If we can't answer it, the paper becomes two disconnected findings glued together.

**Root cause:** the prior study measured preference as a single-dimensional "overall liking" judgment, collected *during* the interaction. This captures in-the-moment comfort but not the participant's sense of downstream value — the very thing UPF is optimizing for.

## 2. Design Principle

**Don't report subjective and objective as two independent findings.** Design the experiment so that the *gap between them* becomes the contribution. Specifically:

> In the moment, users' preference is driven by dialogue naturalness. Once the generalization consequence is made legible, their preference shifts toward the strategy that actually helps them downstream. Therefore the design implication is not "pick UPF or PAR," it is "build UX that surfaces generalization so user preference and robot performance align."

This reframes a seeming contradiction into a design insight compatible with IJHCS's HCI framing.

## 3. Four Concrete Changes to Study 2

### 3.1 Two-pass strategy ranking: blind vs informed

After all three trials are complete:

**Pass 1 (blind).** Participant ranks DQ / UPF / PAR on *"which conversation felt most natural."*

**Pass 2 (informed).** Experimenter shows the participant their own unseen PSR under each strategy — e.g., "For the kitchen trial, the DQ robot correctly placed 3 of the 8 items you didn't mention; the UPF robot placed 6 of 8; the PAR robot placed 4 of 8." Participant re-ranks on *"which would you want for a robot that keeps living with you and helps with new things you buy."*

**Analysis target.** The distribution of rank shifts (pass-1 → pass-2) is the central quantitative finding about the subjective/objective gap. Even if pass-1 favors PAR and pass-2 favors UPF, the paper has a clean story.

### 3.2 Multi-dimensional per-trial ratings

Replace single-item "overall preference" with ≥4 dimensions rated per trial:

| Dimension | Hypothesis |
|---|---|
| Conversation naturalness | PAR likely wins |
| Felt-understood (the robot got my preferences) | UPF likely wins |
| Confidence about new/unseen items (projected generalization) | UPF wins *if* users sense it; key diagnostic |
| Willingness to use long-term | Tracks projected generalization |

The "confidence about new items" rating is the most informative — it directly tests whether users can *spontaneously* detect the generalization benefit, without being told PSR numbers.

### 3.3 Three-layer interview protocol

Semi-structured interview for each strategy, in this order (don't collapse):

1. **Experience layer** — "What made this conversation feel natural or awkward? Walk me through a moment."
2. **Projection layer** — "Imagine this robot stays with you for months and you keep buying new things. How often do you think it would place them right? Why?"
3. **Evidence layer** — (After showing PSR for that trial) "Does seeing this change what you just said? What does?"

The transcripts across these three layers give us a causal chain of the participant's preference revision. Makes the rank shift in §3.1 *explainable*, not just measurable.

### 3.4 Framing for the paper's discussion

Turn the apparent contradiction into a design claim:

- "Users' in-the-moment experience privileges exploratory dialogue; their informed judgment, once generalization is made visible, aligns with the structured-state advantage of UPF."
- "Design implication: the generalization benefit of structured preference state must be *surfaced in the UX* — e.g., the robot explicitly reports the inferences it has made ('Based on what you said, I'd also put X and Y on the bookshelf') — otherwise user preference and robot performance diverge."

This lets the paper land on a constructive recommendation instead of a resigned tradeoff.

## 4. What Stays, What Changes

**Stays (unchanged from current plan):**
- Within-subjects, 3 strategies (DQ / UPF / PAR), B = 6.
- Post-dialogue preference form as ground truth for PSR.
- 4 scenes, Latin square counterbalancing, N ≈ 24.

**Changes / additions:**
- Add multi-dimensional Likert scale after each trial (≥4 items).
- Add blind-vs-informed ranking step after the last trial.
- Expand interview from single "why" question to the three-layer protocol.
- Analysis plan gains a new primary: distribution of rank shifts (Wilcoxon signed-rank on rank change) + Spearman between projected-confidence ratings and actual PSR.

## 5. Open Questions for Next Discussion

Before merging into §6, we need to decide:

1. ~~Should pass-2 informed ranking use the participant's *own* trial-level PSR, or a pooled/average PSR across all participants?~~ **DECIDED: own trial-level PSR only, no pooled context.** Rationale: ground truth is the participant's own preference form, so PSR is inherently personal; the ranking question ("a robot that lives with *you*") is personal; pooled data would dilute the framing. Individual-level noise is absorbed by the rank-shift distribution across N=24.

2. ~~What exactly do we show in the evidence layer?~~ **DECIDED: item-level hits/misses using the participant's own unseen-object names, formatted as a printed card per strategy with all three shown side-by-side during the informed ranking.** Each card lists the items the robot was tested on, ✓/✗ result, and what the participant had indicated. Forces engagement with specific failures, not a "higher is better" glance; also seeds the interview with concrete probe material.

3. ~~Do we disclose which trial used which strategy before ranking, or keep labels abstract?~~ **DECIDED: label by trial order ("第一轮对话 / 第二轮 / 第三轮").** Trial order is the most natural memory anchor; Latin-square counterbalancing neutralizes population-level order effects. Strategy names (DQ/UPF/PAR) are never shown to participants during data collection — revealed only in debrief. The interviewer uses mechanism-descriptive recall ("the one where the robot asked about each object one by one") without ever naming a strategy.

4. ~~Interview vs informed-ranking ordering?~~ **DECIDED: full procedure ordering is —**
   - After each trial: preference form → 4-dim quick ratings (immediate).
   - After all 3 trials, in this exact order:
     1. Blind ranking (pass 1, trial-order labels).
     2. Interview Layer 1 (experience, per-trial).
     3. Interview Layer 2 (projection, per-trial — used for calibration analysis against actual PSR).
     4. Reveal item-level PSR cards (all three side-by-side).
     5. Informed ranking (pass 2) — done before Layer 3 to avoid rationalization contaminating the rank shift.
     6. Interview Layer 3 (evidence) — probe the observed shift or non-shift.
   Rationale: all pre-evidence data (pass-1 ranking, Layers 1 & 2) must be collected before the cards; informed ranking precedes Layer 3 so participants rerank by intuition before explaining themselves.

5. ~~Projected-confidence rating: Likert or forecast count?~~ **DECIDED: concrete forecast count, anchored to actual trial size (0–8, matching the 8 unseen items per trial).** Asked after the preference form, before the next trial. Enables calibration analysis (|predicted − actual|) per strategy — if users systematically underestimate UPF's unseen performance, that directly supports the "generalization benefit is invisible" thesis. Per-trial quick ratings are now: naturalness (Likert 1–7), felt-understood (Likert 1–7), forecast count (0–8), long-term-use willingness (Likert 1–7).

6. ~~N = 24 power check.~~ **DECIDED: keep N=24, no expansion.** Rationale: rank-shift effects (qualitative reappraisal) are typically medium-large when present; null results are theoretically meaningful ("evidence alone doesn't shift preference → UX must actively scaffold"). Two additions: (a) run a power simulation before IRB submission to pre-arm reviewer objections; (b) analysis plan locked to reporting effect sizes (rank-biserial or Cliff's delta) with CIs, not just p-values. If pilot (first 5 participants) shows unusual noise, revisit.

7. ~~Preregistration?~~ **DECIDED: no preregistration.** Analysis plan is still locked in this document (effect sizes, tests, exclusions) and in the final `study2_experiment_guide.md`, giving us an internal timestamp without the external commitment. If a reviewer later challenges post-hoc framing, this doc + commit history serve as the audit trail.

---

## 6. Final Integrated Design (All Q&A Resolved)

### 6.1 Per-trial procedure (run once for each of the 3 strategies)

1. Scene intro (固定).
2. Dialogue phase — participant converses with the robot under one strategy, B = 6, trial-order label only ("第一轮对话" etc.).
3. Post-dialogue preference form — participant assigns every visible object to a receptacle. Serves as this participant's *personal ground truth*.
4. Per-trial quick ratings (four items):
   - Conversation naturalness — Likert 1–7.
   - Felt-understood — Likert 1–7.
   - Forecast count — "Out of the 8 items you didn't mention, how many do you think the robot placed correctly?" (0–8 integer).
   - Long-term-use willingness — Likert 1–7.
5. Short break; proceed to next trial.

### 6.2 Post-session procedure (after all 3 trials)

All steps use trial-order labels. Strategy names are never revealed before debrief.

1. **Blind ranking (pass 1).** Rank the three trials on *"which conversation felt most natural."*
2. **Interview Layer 1 — Experience.** Per trial: "What made this feel natural / awkward? Walk me through a specific moment."
3. **Interview Layer 2 — Projection.** Per trial: "Imagine this robot lives with you for months; how many out of 8 new items do you think it'll get right? Why?" (qualitative reasoning on top of the per-trial forecast count).
4. **Reveal — PSR cards.** Three printed cards laid out side-by-side. Each card: trial-order label, list of the participant's 8 unseen objects for that trial with ✓/✗ and what they had assigned vs what the robot inferred. No percentages as headline numbers.
5. **Informed ranking (pass 2).** Rank on *"which would you want for a robot that lives with you long-term and helps place new things you buy."*
6. **Interview Layer 3 — Evidence.** Probe observed rank shifts (or non-shifts): "Why did X move up? Why did Y stay put even though its card looks better?"
7. **Debrief.** Reveal DQ/UPF/PAR mapping, explain study purpose, collect any final comments.

### 6.3 Confirmatory analyses (locked)

All tests report p-values *and* effect sizes with 95% CIs.

1. **H1 — Rank shift.** Wilcoxon signed-rank on UPF's rank change (pass-2 minus pass-1). Effect size: rank-biserial. Direction: UPF's pass-2 rank is lower (better) than pass-1.
2. **H2 — Calibration gap.** For each participant × strategy, compute |forecast − actual unseen PSR count|. Paired Wilcoxon on the gap for UPF vs DQ, and UPF vs PAR. Effect size: Cliff's delta. Direction: gap is larger for UPF (users underestimate UPF's generalization).
3. **H3 — PSR replication.** Wilcoxon signed-rank on unseen PSR, UPF vs DQ. Direction: UPF > DQ (replicates Study 1 under human conditions).

### 6.4 Exploratory analyses

- Spearman correlation between per-trial forecast count and actual unseen PSR (within-participant and pooled).
- Full Likert distributions across the four per-trial dimensions × three strategies, visualized side-by-side.
- Thematic analysis of Layer 1–3 interview transcripts. Two coders, κ reported.
- Sub-group contrast: rank-shifters vs non-shifters, what differentiates them in Layer 3?

### 6.5 Sample size & exclusions

- N = 24, fixed; Latin-square counterbalanced across 6 orders (4 per order).
- Exclude a participant if: fewer than 3 trials completed, preference form blank for >25% of items, or fails a mid-session attention check (TBD design).
- Run a power simulation *before* IRB submission: assume pass-1 uniform random over the 3 strategies; pass-2 aligned to actual PSR at rates {40%, 50%, 60%, 70%}; report detectable effect at N=24 for each rate. File the simulation script + figure in this notes folder.

### 6.6 What to fold into `study2_experiment_guide.md` next

Once the design above is fully agreed:

- Update the procedure section with §6.1 and §6.2 of this document.
- Add the per-trial rating form and post-session ranking form as appendix artifacts (also translate to Chinese for the instrument).
- Update the interview guide to the three-layer protocol.
- Add §6.3/§6.4/§6.5 to the analysis plan.
- Update §6 of `main.tex` accordingly, and draft §7.5 discussion around the rank-shift framing.

---

## 7. Decision Log

| # | Question | Decision |
|---|---|---|
| Q1 | PSR source for informed ranking | Own trial-level, no pooled |
| Q2 | Evidence layer presentation | Item-level hit/miss printed cards, three side-by-side |
| Q3 | Strategy labels during data collection | Trial-order only; no strategy names until debrief |
| Q4 | Ordering of rankings and interview layers | Per-trial Likert → pass-1 → L1 → L2 → reveal → pass-2 → L3 |
| Q5 | Projection-confidence measure | Forecast count 0–8, not Likert |
| Q6 | Sample size | N=24 retained; add power simulation + report effect sizes |
| Q7 | Preregistration | No external prereg; internal audit trail via this doc |
