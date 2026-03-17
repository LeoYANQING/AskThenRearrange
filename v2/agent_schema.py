from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

try:
    from v2.data import Episode, PlacementMap
except ModuleNotFoundError:
    from data import Episode, PlacementMap


Strategy = Literal["direct", "preference_first", "parallel_exploration"]
QuestionPattern = Literal["action_oriented", "preference_eliciting", "preference_summary"]
PreferenceSource = Literal["elicited", "induced"]
IntentSource = Literal["scene_gap", "action", "induced_hypothesis"]


class QAItem(TypedDict, total=False):
    question_pattern: QuestionPattern
    source: IntentSource
    target: str
    action_mode: Optional[Literal["direct_grounding", "boundary_probe"]]
    question: str
    answer: str


class PreferenceCandidate(TypedDict, total=False):
    preference_id: str
    hypothesis: str
    source: PreferenceSource
    covered_objects: List[str]
    target_receptacle: Optional[str]
    exceptions: List[str]


class ConfirmedPreference(TypedDict, total=False):
    preference_id: str
    hypothesis: str
    source: PreferenceSource
    covered_objects: List[str]
    target_receptacle: Optional[str]
    exceptions: List[str]


class OpenPreferenceDimensionsState(BaseModel):
    dimensions: List[str] = Field(
        description=(
            "A list of high-level preference dimensions worth directly asking "
            "about in the current visible scene."
        )
    )


class PreferenceElicitingIntent(BaseModel):
    source: Literal["scene_gap"] = "scene_gap"
    question_pattern: Literal["preference_eliciting"] = "preference_eliciting"
    dimension: str = Field(description="A concise label for the missing preference dimension worth asking about.")
    covered_objects: List[str] = Field(
        description="Seen objects likely affected by this preference-eliciting intent."
    )
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A direct preference-eliciting question for this dimension.")


class PreferenceElicitingIntentBatch(BaseModel):
    intents: List[PreferenceElicitingIntent] = Field(
        description="A small conservative list of preference-eliciting intents."
    )


class ActionIntent(BaseModel):
    source: Literal["action"] = "action"
    question_pattern: Literal["action_oriented"] = "action_oriented"
    action_mode: Literal["direct_grounding", "boundary_probe"] = "direct_grounding"
    object_name: str = Field(description="One exact unresolved seen object name to ask about next.")
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A direct action-oriented question for this object.")


class PreferenceSummaryIntent(BaseModel):
    source: Literal["induced_hypothesis"] = "induced_hypothesis"
    question_pattern: Literal["preference_summary"] = "preference_summary"
    hypothesis: str = Field(description="A short rule hypothesis to summarize and confirm.")
    covered_objects: List[str] = Field(
        description="Seen objects plausibly covered by this summary hypothesis."
    )
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A direct preference-summary question for this hypothesis.")


class PreferenceSummaryIntentBatch(BaseModel):
    intents: List[PreferenceSummaryIntent] = Field(
        description="A small conservative list of preference-summary intents."
    )


PreferenceIntent = PreferenceElicitingIntent | PreferenceSummaryIntent


class AgentState(TypedDict):
    # control
    strategy: Strategy
    budget_total: int
    budget_used: int

    # task input
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]
    gt_seen_placements: PlacementMap
    gt_unseen_placements: PlacementMap
    annotator_notes: List[str]

    # evidence collected through interaction
    qa_history: List[QAItem]
    asked_questions: List[str]

    # Confirmed seen object -> receptacle actions collected so far.
    confirmed_actions: PlacementMap

    # Preference dimensions not yet acquired, but worth asking directly.
    open_preference_dimensions: List[str]

    # Induced but not yet confirmed preference rules.
    preference_candidates: List[PreferenceCandidate]

    # Preference rules that have already been explicitly confirmed.
    confirmed_preferences: List[ConfirmedPreference]

    # Seen objects whose placement is still unresolved.
    unresolved_objects: List[str]

    # Current turn transient fields.
    current_pattern: Optional[QuestionPattern]
    current_question: Optional[str]
    current_answer: Optional[str]

    # Running prediction outputs.
    predicted_placements_seen: PlacementMap
    predicted_placements_unseen: PlacementMap


__all__ = [
    "ActionIntent",
    "AgentState",
    "ConfirmedPreference",
    "Episode",
    "IntentSource",
    "OpenPreferenceDimensionsState",
    "PlacementMap",
    "PreferenceElicitingIntent",
    "PreferenceElicitingIntentBatch",
    "PreferenceIntent",
    "PreferenceCandidate",
    "PreferenceSource",
    "PreferenceSummaryIntent",
    "PreferenceSummaryIntentBatch",
    "QAItem",
    "QuestionPattern",
    "Strategy",
]
