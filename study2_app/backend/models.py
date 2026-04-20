from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Request models ──────────────────────────────────────────

class CreateSessionInput(BaseModel):
    participant_id: str
    latin_square_row: int  # 1–6
    notes: str = ""
    budget_total: int = 6  # per-trial question budget


class TrialStartInput(BaseModel):
    room_type: str       # "living room" | "bedroom" | "kitchen"
    episode_index: int   # 0–33


class SubmitAnswerInput(BaseModel):
    answer: str


class PreferenceFormInput(BaseModel):
    assignments: Dict[str, str]   # item_name -> receptacle_name


class FinalInput(BaseModel):
    strategy_ranking: List[str]   # e.g. ["UPF", "DQ", "PAR"]
    comment: str = ""


# ── Response / snapshot models ──────────────────────────────

class QATurn(BaseModel):
    turn_index: int
    pattern: str
    question: str
    answer: str = ""
    # Cumulative snapshot of state *after* this turn (experimenter monitor).
    state_after: Optional[Dict[str, Any]] = None


class TrialSnapshot(BaseModel):
    trial_index: int
    strategy: str            # display name: DQ | UPF | PAR
    room_type: str
    episode_index: int
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]
    name_mapping: Dict[str, str] = {}
    dialogue: List[QATurn]
    turns_used: int
    stop_reason: Optional[str]         # "auto_satisfied" | "user_terminated"
    preference_assignments: Optional[Dict[str, str]]
    predicted_placements: Optional[Dict[str, str]]
    psr: Optional[Dict[str, Any]]      # seen_psr, unseen_psr, total_psr, item_scores
    phase: str                         # trial-level phase


class SessionSnapshot(BaseModel):
    session_id: str
    participant_id: str
    latin_square_row: int
    trial_order: List[Dict[str, str]]  # [{strategy, room_type}, ...]
    current_trial_index: int
    trials: List[TrialSnapshot]
    phase: str                         # global phase
    notes: str
    agent_state: Optional[Dict[str, Any]] = None  # live serialized AgentState
    strategy_ranking: Optional[List[str]] = None  # set after final_ranking submission
    final_comment: str = ""
    budget_total: int = 6


class NextQuestionResponse(BaseModel):
    question: str
    pattern: str
    turn_index: int
    dialogue_complete: bool


class ScoreResponse(BaseModel):
    seen_psr: float
    unseen_psr: float
    total_psr: float
    item_scores: Dict[str, bool]
