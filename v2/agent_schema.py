from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

try:
    from v2.data import Episode
except ModuleNotFoundError:
    from data import Episode


Strategy = Literal["direct", "preference_first", "parallel_exploration"]
QuestionPattern = Literal["action_oriented", "preference_eliciting", "preference_summary"]


class QAItem(TypedDict, total=False):
    question_pattern: QuestionPattern
    target: str
    action_mode: Optional[Literal["direct_grounding", "boundary_probe"]]
    question: str
    answer: str


class LearnedPreference(TypedDict, total=False):
    hypothesis: str
    covered_objects: List[str]

class ActionRecord(TypedDict):
    object_name: str
    receptacle: str

class AgentState(TypedDict):
    budget_total: int

    # task input
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]

    # evidence collected through interaction
    qa_history: List[QAItem]
    # Confirmed seen object -> receptacle actions collected so far.
    confirmed_actions: List[ActionRecord]
    # Object -> receptacle actions the user explicitly ruled out.
    negative_actions: List[ActionRecord]

    # Confirmed user preferences learned so far.
    confirmed_preferences: List[LearnedPreference]
    # Preferences the user explicitly rejected.
    negative_preferences: List[LearnedPreference]
    # Seen objects whose placement is still unresolved.
    unresolved_objects: List[str]


__all__ = [
    "AgentState",
    "Episode",
    "ActionRecord",
    "LearnedPreference",
    "QAItem",
    "QuestionPattern",
]
