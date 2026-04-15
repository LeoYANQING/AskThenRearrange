# Study 2 User Experiment Guide

## 1. Purpose

Study 2 evaluates whether the strategy effects observed in simulation also appear with real users, and whether objective performance aligns with subjective interaction quality.

## 2. Research Questions

- **RQ1**: To what extent do strategy-dependent differences in placement generalization replicate when the agent interacts with real users rather than simulated oracles?
- **RQ2**: How do questioning strategies shape users' perceived workload, perceived control, and overall interaction quality during preference teaching?
- **RQ3**: Do objective task performance and users' strategy preferences diverge, and if so, what kinds of trade-offs characterize that divergence?

## 3. Design

- **Design**: within-subjects
- **Conditions**: DQ, UPF, PAR
- **Counterbalancing**: Latin-square style assignment for strategy order; scene order should be counterbalanced separately rather than fixed to trial position
- **Question budget**: fixed at `B = 6`
- **Sample target**: `N = 20-24`

### Strategy Conditions

| Code | Strategy | Patterns |
|---|---|---|
| DQ | Direct Querying | AO only |
| UPF | User-Preference-First | PE -> AO |
| PAR | Parallel Exploration | AO -> PI |

## 4. Core Design Decision: Preference Form After Dialogue

The preference form should be completed **after** the dialogue and **before** the system reveals its final placement plan.

### Rationale

- If participants fill in a full placement sheet before the interaction, they become highly familiar with the object set and receptacles in advance.
- That pre-exposure would reduce ecological validity and make workload comparisons across strategies harder to interpret.
- Post-dialogue elicitation better preserves the natural teaching situation: participants first answer the agent's questions, then articulate their own full preference structure for evaluation.

### Consequence for Evaluation

- The post-dialogue form functions as the participant's **reference preference annotation** for that trial.
- It should be collected before any system prediction is shown, so the annotation remains independent of system output.
- In the paper, this should be described carefully as a post-interaction ground-truth elicitation step rather than a pre-existing gold label.

## 5. Trial Structure

Each participant completes three trials, one per strategy, using counterbalanced household scenes drawn from a four-scene pool.

### Per-Trial Flow

1. Present the scene and receptacle layout.
2. Run the dialogue with the assigned strategy (`B = 6` questions).
3. Ask the participant to complete a preference form assigning all items to receptacles.
4. Show the system's final placement plan.
5. Administer the post-trial questionnaire.

### Timing

- Consent + demographics + ATI/NARS: `5-8 min`
- Training / practice: `5 min`
- Each trial: `12-15 min`
- Final ranking + interview: `10-15 min`
- Total: about `60-70 min`

## 6. Scenes

Use four matched household scenes with comparable complexity.

### Constraints

- `12` items per scene
- `5` receptacles per scene
- `3-4` plausible semantic categories
- at least a few boundary cases per scene
- similar ambiguity level across scenes

### Candidate Scenes

- study
- bedroom
- kitchen
- living room

The exact item lists should be maintained in a separate materials sheet rather than embedded in this guide.

## 7. Measures

### Objective Measures

- **Placement accuracy** over all items, relative to the participant's post-dialogue preference form
- **Discussed-item accuracy**: accuracy on items explicitly mentioned during dialogue
- **Undiscussed-item accuracy**: accuracy on items not explicitly mentioned during dialogue

Note: because the preference form is collected after dialogue, these measures should be framed as agreement with participants' articulated post-dialogue preferences, not recovery of a pre-registered hidden ground truth.

### Subjective Measures

- NASA-TLX
- PSC
- perceived control
- overall preference ranking across strategies

### Behavioral Measures

- number of turns
- pattern counts (AO / PE / PI)
- response length
- repair events, clarification requests, parsing failures

### Qualitative Data

- short semi-structured exit interview

## 8. Participants

### Inclusion Criteria

- age `18+`
- regular experience organizing household items
- technically diverse sample
- no participation in the earlier study

### Background Measures

- age, gender, education
- ATI
- NARS
- self-reported organizing frequency and organizing confidence

## 9. Analysis Plan

### Main Quantitative Analysis

Use repeated-measures models with strategy as the main within-subject factor.

Primary comparisons:

- UPF vs DQ on undiscussed-item accuracy
- PAR vs DQ on undiscussed-item accuracy

### Subjective Analysis

- NASA-TLX by strategy
- PSC by strategy
- perceived control by strategy
- preference ranking with non-parametric repeated-measures tests if needed

### Interpretation Goal

The main theoretical target is not only whether one strategy is better, but whether different strategies optimize different aspects of the interaction:

- generalization accuracy
- cognitive burden
- perceived intelligence or curiosity
- felt control

## 10. Interview Focus

Keep the exit interview short and comparative.

Suggested prompts:

- Which interaction style felt most natural, and why?
- Which style best captured your preferences?
- Which style felt most effortful?
- Were any questions confusing, redundant, or presumptuous?
- Would you want the system to keep one fixed style or adapt its style over time?

## 11. Open Decisions

- Whether 3 trials per participant is sufficient, or whether a fourth scene should be added in a future revision
- Whether the final interface is robot-based or screen-based
- Whether discussed / undiscussed is sufficient, or whether a stronger seen / unseen manipulation should be introduced in a later revision

## 12. Recommended Paper Framing

When writing the paper, position Study 2 as a test of whether strategy effects remain meaningful under real-user variability, where answers may be ambiguous, incomplete, or differently granular from the simulated oracle in Study 1.
