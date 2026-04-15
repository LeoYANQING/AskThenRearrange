from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired


PolicyMode = Literal[
    "direct_querying",
    "user_preference_first",
    "parallel_exploration",
]

TrialStatus = Literal[
    "assigned",
    "dialogue_active",
    "dialogue_waiting_for_answer",
    "dialogue_processing",
    "dialogue_failed",
    "dialogue_complete",
    "preference_form_active",
    "results_computed",
    "questionnaire_pending",
    "trial_complete",
    "trial_interrupted",
]

ParticipantStatus = Literal["active", "completed", "interrupted"]


class TrialAssignment(TypedDict):
    trial_id: str
    trial_index: int
    strategy: PolicyMode
    scene_id: str
    scene_label: str
    scene_episode_index: int


class PendingTurnRecord(TypedDict):
    turn_index: int
    question_pattern: str
    guidance: str
    target: str
    question: str
    action_mode: str | None
    hypothesis: str
    covered_objects: list[str]
    receptacle: str | None
    created_at: str
    retry_count: int


class TurnRecord(TypedDict):
    turn_index: int
    question_pattern: str
    guidance: str
    target: str
    question: str
    action_mode: str | None
    covered_objects: list[str]
    participant_answer_raw: str
    update_status: str
    retry_count: int
    state_summary: dict[str, Any]
    created_at: str
    answered_at: str


class FailureRecord(TypedDict):
    stage: str
    message: str
    turn_index: int
    failed_at: str
    retry_count: int
    pending_turn: PendingTurnRecord | None
    answer_text: str
    state_before_turn: dict[str, Any]


class PreferenceFormRecord(TypedDict):
    trial_id: str
    placements: dict[str, str]
    submitted_at: str


class TrialResultRecord(TypedDict):
    trial_id: str
    predicted_placements: dict[str, str]
    participant_reference_placements: dict[str, str]
    discussed_items: list[str]
    undiscussed_items: list[str]
    discussed_item_accuracy: float
    undiscussed_item_accuracy: float
    overall_accuracy: float
    confirmed_actions_summary: list[dict[str, str]]
    confirmed_preferences_summary: list[dict[str, Any]]
    computed_at: str


class TrialRecord(TypedDict):
    trial_id: str
    participant_id: str
    trial_index: int
    strategy: PolicyMode
    scene_id: str
    scene_label: str
    scene_episode_index: int
    budget_total: int
    questionnaire_url: str
    status: TrialStatus
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    agent_state: dict[str, Any] | None
    pending_turn: PendingTurnRecord | None
    turns: list[TurnRecord]
    failure: FailureRecord | None
    failure_history: list[FailureRecord]
    preference_form: PreferenceFormRecord | None
    result: TrialResultRecord | None


class ParticipantRecord(TypedDict):
    participant_id: str
    counterbalance_group: str
    strategy_group: int
    scene_group: int
    trial_order: list[str]
    current_trial_index: int
    status: ParticipantStatus
    created_at: str
    updated_at: str
    completed_at: str | None
    notes: NotRequired[str]
