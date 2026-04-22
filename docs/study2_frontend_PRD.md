# Study 2 Frontend System — Product Requirements Document

**Project**: AskThenRearrange / PrefQuest  
**Document Version**: 1.0  
**Date**: 2026-04-17  
**Environment**: conda `behavior`  
**Stack**: FastAPI (Python) + React (TypeScript) + OpenAI TTS / Whisper API

---

## 1. Overview

This document specifies the frontend system for **Study 2** of the PrefQuest project. The system enables a human experimenter to run within-subjects user study sessions in which participants teach a simulated household-arrangement agent their preferences through spoken dialogue. The agent uses one of three questioning strategies (DQ / UPF / PAR) per trial; the system records full behavioural, objective, and subjective data for later analysis.

### 1.1 Study Context

- **Design**: Within-subjects, 3 strategy conditions (DQ, UPF, PAR)
- **Budget**: open-ended; participant self-terminates by verbal signal (no fixed turn cap; system enforces a soft maximum of 20 turns as a safety backstop)
- **Participants**: N = 24
- **Scenes**: 4 household scenes (study / bedroom / kitchen / living room), each with 16 items (8 seen + 8 unseen) and 5 receptacles
- **Counterbalancing**: Latin-square order over strategies; scene assignment counterbalanced separately
- **Session length**: ~60–70 min
- **Physical setup**: Real physical tabletop mock-scenes; experimenter manually enters scene content into the system before each session

### 1.2 Interaction Model

1. Experimenter configures the scene for a trial (items + receptacles + scene label)
2. Agent asks questions (text displayed + TTS audio played); participant answers verbally (Whisper STT); participant self-terminates by verbal signal when satisfied
3. Participant fills in a preference form (all 16 items → receptacle assignments) — **before any prediction is shown**
4. Participant completes per-trial questionnaire (CL / PU / IA primary; QA process check) — **blind to placement accuracy**
5. System displays its predicted placement plan alongside participant's preference form (matches/mismatches highlighted)
6. After all 3 trials, participant completes a final strategy preference ranking and optional comment box
7. All data is written to a JSONL session log

---

## 2. Architecture

```
┌──────────────────────────────────────┐
│             React Frontend           │
│  ExperimenterDashboard │ ParticipantUI│
│  SceneSetup │ DialogueView │ Forms   │
└──────────────┬───────────────────────┘
               │  HTTP / WebSocket
┌──────────────▼───────────────────────┐
│           FastAPI Backend            │
│  /session  /dialogue  /evaluate      │
│  /tts  /stt  /log  /export           │
├──────────────────────────────────────┤
│  PrefQuest Core (existing Python)    │
│  question_policy.py  proposers.py    │
│  state_update.py  evaluation.py      │
│  llm_factory.py  oracle.py           │
├──────────────────────────────────────┤
│  OpenAI API                          │
│  tts-1 (speech output)               │
│  whisper-1 (speech input)            │
│  LLM backend (GPT-5-chat default)    │
└──────────────────────────────────────┘
```

### 2.1 Directory Structure

```
study2_app/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── routers/
│   │   ├── session.py           # Session lifecycle
│   │   ├── dialogue.py          # Turn-by-turn question/answer
│   │   ├── evaluation.py        # PSR computation
│   │   ├── tts.py               # Text-to-speech proxy
│   │   ├── stt.py               # Speech-to-text proxy
│   │   └── log.py               # Log writing & export
│   ├── models.py                # Pydantic request/response models
│   ├── session_store.py         # In-memory session state
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ExperimenterDashboard.tsx
│   │   │   ├── SceneSetup.tsx
│   │   │   ├── DialoguePage.tsx
│   │   │   ├── PreferenceForm.tsx
│   │   │   ├── SystemPrediction.tsx
│   │   │   ├── Questionnaire.tsx
│   │   │   └── FinalRanking.tsx
│   │   ├── components/
│   │   │   ├── AudioPlayer.tsx
│   │   │   ├── VoiceRecorder.tsx
│   │   │   ├── ItemReceptacleGrid.tsx
│   │   │   ├── NasaTLX.tsx
│   │   │   └── PSCScale.tsx
│   │   ├── api.ts               # Backend API client
│   │   └── App.tsx
│   └── package.json
└── logs/                        # JSONL session logs (output)
```

---

## 3. Experimenter Dashboard

### 3.1 Session Setup

The experimenter creates a new session before the participant arrives.

**Fields**:

| Field | Type | Description |
|---|---|---|
| `participant_id` | string | e.g. `P07` |
| `session_date` | date | auto-filled to today |
| `latin_square_row` | int (1–6) | selects strategy + scene order for this participant |
| `notes` | text | optional freeform |

**Latin-square table** (hard-coded, editable in config):

| Row | Trial 1 | Trial 2 | Trial 3 |
|---|---|---|---|
| 1 | DQ / study | UPF / bedroom | PAR / kitchen |
| 2 | DQ / bedroom | PAR / study | UPF / living |
| 3 | UPF / kitchen | DQ / living | PAR / study |
| 4 | UPF / living | PAR / bedroom | DQ / kitchen |
| 5 | PAR / study | DQ / kitchen | UPF / bedroom |
| 6 | PAR / kitchen | UPF / study | DQ / living |

The default table above is a starting point; the experimenter can override per-trial strategy/scene assignment via the dashboard.

### 3.2 Scene Configuration (per trial)

Before each trial the experimenter enters the physical scene content via a scene setup form.

**Fields**:

| Field | Description |
|---|---|
| `scene_label` | e.g. `bedroom` |
| `receptacles` | 5 × { id, display_name } e.g. `dresser`, `nightstand`, `closet`, `shelf`, `under-bed` |
| `items` | 12 × { id, display_name, default_category (optional hint) } |
| `strategy` | auto-filled from Latin-square row; experimenter can override |
| `budget` | default 6; editable |

Items and receptacles are saved into the trial's `AgentState` before the dialogue begins.

### 3.3 Session Control Panel

A persistent side panel (visible only on experimenter view, not participant view) shows:

- Current participant ID, trial number, strategy
- Budget remaining
- A **"Advance"** button to move between phases (scene → dialogue → pref form → prediction → questionnaire)
- An **emergency stop / reset trial** button (confirms before acting)
- Live log tail (last 5 events)

---

## 4. Participant UI

The participant UI is displayed on a second screen (or browser window/tab) and advances only when the experimenter clicks "Advance" or when the system automatically progresses (e.g., after audio playback ends).

### 4.1 Phase 1 — Scene Introduction

Displays:
- Scene name and a brief framing sentence (e.g., "Imagine this is your bedroom. Please look at the items on the table.")
- A visual grid showing the 12 items and 5 receptacles (icon + text label)
- A "Ready" button the participant presses when they have reviewed the scene

### 4.2 Phase 2 — Dialogue

Each turn:

1. **Question display**: The agent's question appears as a chat bubble.
2. **TTS playback**: Audio is automatically played (using OpenAI TTS).
3. **Answer capture**:
   - A microphone button lets the participant record their answer (Whisper STT).
   - A text field is also shown as fallback; the participant can edit the transcription before submitting.
4. **Submit**: Participant presses "Submit Answer" (or hits Enter).
5. Budget counter (e.g., "Question 3 / 6") is shown.
6. After B=6 turns, dialogue ends automatically; a "Dialogue complete" message is shown.

**STT flow**:
- Press-and-hold to record (up to 30 s)
- Release to send audio to backend → Whisper → transcription returned within ~2 s
- Transcription appears in the text field; participant can correct
- Submit sends the (possibly edited) text

### 4.3 Phase 3 — Preference Form

Displayed after dialogue, before system prediction.

- Instruction: "Now please assign each item to the receptacle where you would most prefer it to be stored."
- A drag-and-drop grid: items on the left, receptacles as columns/buckets.
- Each item must be assigned to exactly one receptacle before submission.
- A "Submit Preferences" button becomes active once all 12 items are assigned.
- This form response is stored as the **ground-truth reference** for PSR evaluation.

### 4.4 Phase 4 — System Prediction

Displayed after the preference form is submitted.

- The system's predicted placement for all 12 items is shown (same grid layout, but now showing agent's choices).
- Matches and mismatches with the participant's form are highlighted (green / red).
- A summary line: "The agent correctly placed X / 12 items according to your preferences."
- Participant reads this, then signals they are ready for the questionnaire.

### 4.5 Phase 5 — Per-Trial Questionnaire

> **Authoritative questionnaire source**: [`docs/study2_session_sop.md`](study2_session_sop.md) §Phase 4. The SOP reflects the current v2.3 (2026-04-21) instrument design, which supersedes the earlier NASA-TLX / PSC / Perceived Control scheme outlined below. Frontend implementers MUST render the SOP items verbatim (CL1–3, PU1–3, IA1–3, QA1–2).
>
> **Measurement tiers** (v2.3):
> - **Primary** (hypothesis-testing, Friedman + post-hoc): CL, PU, IA — map to RQ4 / H3.
> - **Process check** (no hypothesis test; descriptive + Spearman(QA, PU)): QA.
> - **Behavioral outcome**: Final Strategy Ranking in §4.6 (replaces a Likert Trust & Intention subscale).

Four instruments on a single scrollable page (all 7-point Likert unless noted):

#### CL — Cognitive Load (3 items) — *primary*

Anchors: 1 = very low / 7 = very high. See SOP for item wording.

#### PU — Perceived Understanding (3 items) — *primary*

Anchors: 1 = strongly disagree / 7 = strongly agree. See SOP.

#### IA — Interaction Agency (3 items) — *primary*

Anchors: 1 = strongly disagree / 7 = strongly agree. See SOP.

#### QA — Questioning Appropriateness (2 items) — *process check, not hypothesis-tested*

Anchors: 1 = strongly disagree / 7 = strongly agree.

1. The way the agent asked its questions felt natural for this task.
2. I could easily understand why the agent was asking each question.

> Rationale: QA probes whether HHI-derived questioning patterns, once operationalized as agent-side strategies, are perceived as appropriate by real users. Analysis plan reports descriptive statistics and Spearman(QA, PU) only; QA is excluded from H1–H3 family-wise corrections to keep the primary story focused on CL/PU/IA.

A "Submit" button finalizes the trial. After submission the experimenter advances to the next trial or the final session.

### 4.6 Phase 6 — Final Session (after all 3 trials)

#### Strategy Preference Ranking

- Three strategy labels are presented (system may show brief one-sentence descriptions).
- Participant drag-ranks them 1st / 2nd / 3rd.

#### Final Open Comment Box

- "Is there anything else you'd like to share about your experience?"
- Freeform text, optional.

#### Demographics & Background (collected once, at start of session)

- Age (numeric)
- Gender (free text + prefer-not-to-say option)
- Education level (dropdown)
- Organizing frequency (5-point: rarely → daily)
- Organizing confidence (5-point)
- ATI scale (9 items, 6-point Likert) — short form
- NARS scale (9 items, 5-point Likert) — S1+S2 subscales

---

## 5. Backend API

All endpoints are prefixed `/api/v1`.

### 5.1 Session Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/session/create` | Create new session; returns `session_id` |
| GET | `/session/{session_id}` | Get current session state |
| POST | `/session/{session_id}/trial/start` | Start a new trial (supply scene config + strategy) |
| POST | `/session/{session_id}/trial/end` | Finalize current trial |
| POST | `/session/{session_id}/end` | End session, flush logs |

### 5.2 Dialogue Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/dialogue/{session_id}/next_question` | Generate next question from PrefQuest; returns `{question_text, pattern, budget_remaining}` |
| POST | `/dialogue/{session_id}/submit_answer` | Submit participant answer (text); triggers state update; returns `{updated_state}` |

### 5.3 Evaluation Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/evaluate/{session_id}/predict` | Run placement prediction; returns `{predicted_placements}` |
| POST | `/evaluate/{session_id}/score` | Compute PSR given preference form + predicted placements; returns `{seen_psr, unseen_psr, total_psr, item_scores}` |

### 5.4 Speech Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/tts` | Body: `{text, voice?}`; returns audio/mpeg stream (OpenAI TTS `tts-1` model, voice default `alloy`) |
| POST | `/stt` | Body: multipart/form-data with `audio` file; returns `{transcript}` (OpenAI Whisper `whisper-1`) |

### 5.5 Log Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/log/{session_id}` | Retrieve current session log as JSONL |
| GET | `/log/{session_id}/export` | Download complete session log as `.jsonl` file |

---

## 6. Data Model

### 6.1 Session Log Schema (JSONL)

Each line is a JSON object with a `type` field. Compatible with the existing `logs/` directory format.

```jsonc
// session_start
{
  "type": "session_start",
  "session_id": "P07_20260417",
  "participant_id": "P07",
  "latin_square_row": 3,
  "timestamp": "2026-04-17T14:00:00Z"
}

// trial_start
{
  "type": "trial_start",
  "session_id": "P07_20260417",
  "trial_index": 0,
  "strategy": "UPF",
  "scene_label": "bedroom",
  "budget": 6,
  "receptacles": ["dresser","nightstand","closet","shelf","under-bed"],
  "items": ["book","phone charger","glasses","laptop","notebook", ...],
  "timestamp": "..."
}

// dialogue_turn
{
  "type": "dialogue_turn",
  "session_id": "P07_20260417",
  "trial_index": 0,
  "turn_index": 0,
  "pattern": "PE",
  "question": "How do you generally decide where to put things in your bedroom?",
  "answer": "I usually put things I use every night on the nightstand.",
  "stt_raw": "I usually put things I use every night on the nightstand.",
  "answer_edited": false,
  "response_time_s": 8.2,
  "timestamp": "..."
}

// preference_form
{
  "type": "preference_form",
  "session_id": "P07_20260417",
  "trial_index": 0,
  "assignments": {"book": "shelf", "phone charger": "nightstand", ...},
  "timestamp": "..."
}

// evaluation
{
  "type": "evaluation",
  "session_id": "P07_20260417",
  "trial_index": 0,
  "predicted_placements": {"book": "shelf", "phone charger": "nightstand", ...},
  "seen_psr": 0.857,
  "unseen_psr": 0.714,
  "total_psr": 0.833,
  "timestamp": "..."
}

// questionnaire
{
  "type": "questionnaire",
  "session_id": "P07_20260417",
  "trial_index": 0,
  "nasa_tlx": {"mental": 55, "physical": 10, "temporal": 30, "performance": 70, "effort": 45, "frustration": 20},
  "psc": [6, 5, 6, 5, 4],
  "perceived_control": [6, 5, 5],
  "timestamp": "..."
}

// session_end
{
  "type": "session_end",
  "session_id": "P07_20260417",
  "strategy_ranking": ["UPF", "DQ", "PAR"],
  "final_comment": "The second style felt more natural.",
  "demographics": { "age": 28, "gender": "female", "education": "graduate", ... },
  "timestamp": "..."
}
```

---

## 7. PrefQuest Integration

The backend wraps the existing Python modules. No changes to core logic are required.

### 7.1 State Initialization

On `trial/start`, construct an `AgentState` dict:

```python
from agent_schema import AgentState

state: AgentState = {
    "budget_total": 6,
    "room": scene_label,
    "receptacles": receptacle_list,          # list[str]
    "seen_objects": items[:6],               # first 6 = "seen" during dialogue
    "unseen_objects": items[6:],             # last 6 = held out for generalization
    "qa_history": [],
    "confirmed_actions": [],
    "negative_actions": [],
    "confirmed_preferences": [],
    "negative_preferences": [],
    "unresolved_objects": list(items),
}
```

The seen/unseen split is fixed per scene and pre-defined in scene config (see §3.2). The experimenter does not see or control this split; it is an internal evaluation construct.

### 7.2 Question Generation

On `dialogue/next_question`, call:

```python
from question_policy import QuestionPolicyController
controller = QuestionPolicyController()
question_obj = controller.plan_next_question(state, mode=strategy)
# question_obj: { "question": str, "pattern": "AO"|"PE"|"PI" }
```

### 7.3 State Update

On `dialogue/submit_answer`, call:

```python
from state_update import update_state_from_answer
state = update_state_from_answer(state, question_obj["question"], answer_text)
```

### 7.4 Evaluation

On `evaluate/predict`, call:

```python
from evaluation import predict_placements
predicted = predict_placements(state)
# predicted: dict[item_name, receptacle_name]
```

On `evaluate/score`, compute PSR against the participant's preference form:

```python
def compute_psr(predicted: dict, reference: dict, seen: list, unseen: list):
    seen_correct = sum(predicted[i] == reference[i] for i in seen)
    unseen_correct = sum(predicted[i] == reference[i] for i in unseen)
    return {
        "seen_psr": seen_correct / len(seen),
        "unseen_psr": unseen_correct / len(unseen),
        "total_psr": (seen_correct + unseen_correct) / (len(seen) + len(unseen)),
    }
```

### 7.5 TTS Integration

```python
import openai
client = openai.OpenAI()
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",          # configurable: alloy / echo / fable / onyx / nova / shimmer
    input=question_text,
)
# Stream audio bytes back to frontend
```

### 7.6 STT Integration

```python
response = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,        # uploaded from frontend (webm / mp4 / wav)
    language="en",
)
transcript = response.text
```

---

## 8. Configuration

All runtime configuration lives in a single `config.yaml` (or `.env`) file:

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  tts_model: tts-1
  tts_voice: alloy
  stt_model: whisper-1
  llm_model: gpt-5-chat
  llm_base_url: https://api.openai.com/v1

study:
  budget: 6
  scenes:
    - label: bedroom
      seen_items: [book, glasses, phone_charger, notebook, pen, earplugs]
      unseen_items: [lamp, watch, headphones, charger_cable, journal, hand_cream]
      receptacles: [dresser, nightstand, closet, shelf, under_bed]
    - label: kitchen
      ...
    - label: study
      ...
    - label: living_room
      ...
  latin_square:
    - [DQ/bedroom, UPF/kitchen, PAR/study]
    - [DQ/kitchen, PAR/bedroom, UPF/living]
    - [UPF/study, DQ/living, PAR/bedroom]
    - [UPF/living, PAR/kitchen, DQ/study]
    - [PAR/bedroom, DQ/study, UPF/kitchen]
    - [PAR/study, UPF/bedroom, DQ/living]

logging:
  output_dir: logs/
  filename_template: "{participant_id}_{date}.jsonl"
```

---

## 9. Non-Functional Requirements

| Requirement | Target |
|---|---|
| TTS latency | < 2 s from question generation to audio start |
| STT latency | < 3 s from audio upload to transcript return |
| LLM question generation | < 5 s per turn |
| Simultaneous sessions | 1 (single experimenter, single participant) |
| Browser support | Chrome 120+, full-screen mode recommended |
| Audio input | WebM or WAV from browser MediaRecorder API |
| Log durability | Write JSONL line after every event; no data loss on crash |
| Offline recovery | Session state persisted to disk (`session_store.json`) so sessions can resume after network interruption |

---

## 10. Experimenter Workflow (End-to-End)

1. **Pre-session**: Set up physical mock scene; prepare 4 scene config YAML snippets.
2. **Open dashboard**: Navigate to `http://localhost:8000` in experimenter browser.
3. **Create session**: Enter participant ID, select Latin-square row.
4. **Demographics**: Hand participant the screen for demographics + ATI/NARS (Phase 0, one-time).
5. **Practice trial**: Run one practice trial with a sample scene (not counted); experimenter explains each phase.
6. **Trials 1–3**:
   a. Experimenter enters scene config (items + receptacles) for the trial.
   b. Experimenter clicks "Start Dialogue" → participant view auto-advances.
   c. Agent generates Question 1; TTS plays; participant answers via mic or keyboard.
   d. Repeat for B=6 turns.
   e. Experimenter clicks "Advance to Preference Form".
   f. Participant completes preference form on their screen.
   g. Experimenter clicks "Show Prediction".
   h. Participant completes per-trial questionnaire.
   i. Experimenter clicks "Next Trial".
7. **Final session**: Participant completes strategy ranking + comment; experimenter clicks "End Session".
8. **Export**: Download JSONL log file for analysis.

---

## 11. Out of Scope (v1)

- Automatic scene recognition from camera / image
- Multi-participant concurrent sessions
- Automated counterbalancing assignment (Latin-square row is manually entered)
- Integration with existing `test_policy_loop.py` simulation runner
- Mobile / tablet layout

---

## 12. Acceptance Criteria

| # | Criterion |
|---|---|
| AC-01 | Experimenter can create a session, configure a scene (12 items, 5 receptacles), and start a dialogue in < 2 min of setup time |
| AC-02 | Agent generates a question using the correct strategy; pattern label (AO/PE/PI) is logged for every turn |
| AC-03 | TTS audio plays within 2 s of question generation |
| AC-04 | Whisper transcription returns within 3 s of audio submission; participant can edit before submitting |
| AC-05 | Preference form enforces all 12 items assigned before submission |
| AC-06 | Seen PSR, unseen PSR, and total PSR are computed correctly against the participant's preference form |
| AC-07 | JSONL log contains one event line per action (session_start, trial_start, dialogue_turn × B, preference_form, evaluation, questionnaire, session_end) |
| AC-08 | Session state survives a backend restart; experimenter can resume a trial |
| AC-09 | NASA-TLX, PSC, and Perceived Control questionnaires are all correctly captured and logged |
| AC-10 | Final JSONL export downloads correctly and can be loaded with the existing analysis scripts |
