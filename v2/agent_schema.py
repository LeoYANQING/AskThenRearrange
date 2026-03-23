from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

try:
    from v2.data import Episode, PlacementMap
except ModuleNotFoundError:
    from data import Episode, PlacementMap


Strategy = Literal["direct", "preference_first", "parallel_exploration"]
QuestionPattern = Literal["action_oriented", "preference_eliciting", "preference_summary"]
PreferenceSource = Literal["elicited", "induced", "confirmed"]


class QAItem(TypedDict, total=False):
    question_pattern: QuestionPattern
    target: str
    action_mode: Optional[Literal["direct_grounding", "boundary_probe"]]
    question: str
    answer: str


class PreferenceRecord(TypedDict, total=False):
    hypothesis: str
    source: PreferenceSource
    covered_objects: List[str]
    target_receptacle: Optional[str]
    exceptions: List[str]


class PreferenceElicitingIntent(BaseModel):
    question_pattern: Literal["preference_eliciting"] = "preference_eliciting"
    hypothesis: str = Field(description="A concise preference hypothesis worth asking the user to clarify.")
    covered_objects: List[str] = Field(
        description="Seen objects likely affected by this preference-eliciting intent."
    )
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A direct preference-eliciting question for this hypothesis.")


class PreferenceElicitingIntentBatch(BaseModel):
    intents: List[PreferenceElicitingIntent] = Field(
        description="A small conservative list of preference-eliciting intents."
    )


class ActionIntent(BaseModel):
    question_pattern: Literal["action_oriented"] = "action_oriented"
    action_mode: Literal["direct_grounding", "boundary_probe"] = "direct_grounding"
    object_name: str = Field(description="One exact unresolved seen object name to ask about next.")
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A direct action-oriented question for this object.")


class PreferenceSummaryIntent(BaseModel):
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

    # evidence collected through interaction
    qa_history: List[QAItem]

    # Confirmed seen object -> receptacle actions collected so far.
    confirmed_actions: PlacementMap

    # Object -> receptacles explicitly ruled out by prior answers.
    excluded_receptacles: dict[str, List[str]]

    # Preference rules that have already been explicitly confirmed.
    confirmed_preferences: List[PreferenceRecord]

    # Preference hypotheses still worth asking directly.
    open_preference_hypotheses: List[str]

    # Summary hypotheses that have been explicitly rejected.
    rejected_hypotheses: List[str]

    # Induced but not yet confirmed preference rules.
    preference_candidates: List[PreferenceRecord]

    # Online placements derived from confirmed evidence during interaction.
    online_placements_seen: PlacementMap
    
    # Seen objects whose placement is still unresolved.
    unresolved_objects: List[str]


__all__ = [
    "ActionIntent",
    "AgentState",
    "Episode",
    "PlacementMap",
    "PreferenceRecord",
    "PreferenceElicitingIntent",
    "PreferenceElicitingIntentBatch",
    "PreferenceSource",
    "PreferenceSummaryIntent",
    "PreferenceSummaryIntentBatch",
    "QAItem",
    "QuestionPattern",
    "Strategy",
]
