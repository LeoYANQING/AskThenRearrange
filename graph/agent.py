from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict


# ---------- basic enums ----------

QPattern = Literal["action", "preference_elicit", "preference_confirm"]
Strategy = Literal["direct", "preference_first", "parallel_exploration"]
IntentSource = Literal["scene_gap", "induced_hypothesis", "rule_followup"]


# ---------- dataset episode ----------

@dataclass
class Episode:
    episode_id: str
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]

    # Ground truth is only used for oracle answering / evaluation.
    gt_seen_placements: Dict[str, str]
    gt_unseen_placements: Dict[str, str]

    # Oracle-only personalized knowledge. The agent should NOT read this directly.
    annotator_notes: List[str]

    # Optional metadata from the dataset
    tags: List[str]


# ---------- interaction records ----------

class QAItem(TypedDict, total=False):
    turn_id: int
    pattern: QPattern
    source: Optional[IntentSource]
    target: Optional[str]
    question: str
    answer: str


# ---------- preference state ----------

class PreferenceHypothesis(TypedDict):
    hypothesis_id: str
    statement: str
    source: Literal["elicited", "induced"]
    status: Literal["open", "confirmed", "revised"]

    # Which concrete action evidence supports this rule
    supporting_objects: List[str]

    # Which objects this rule is currently believed to cover
    covered_objects: List[str]

    # Optional target receptacle if the rule has a concrete placement target
    target_receptacle: Optional[str]

    # Human-readable edge cases
    exceptions: List[str]


class PreferenceIntent(TypedDict):
    """
    Output schema shared by future proposer modules.
    These are NOT final natural-language questions yet.
    They are candidate motivations/intents that a selector can compare.
    """
    intent_id: str
    source: IntentSource
    recommended_pattern: Literal["preference_elicit", "preference_confirm"]
    focus: str
    motivation: str
    covered_objects: List[str]
    priority: float


# ---------- agent state ----------

class AgentState(TypedDict):
    # control
    strategy: Strategy
    budget_total: int
    budget_used: int
    current_pattern: Optional[QPattern]

    # task input
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    unseen_objects: List[str]

    # oracle-only references; agent nodes should not read annotator_notes directly
    gt_seen_placements: Dict[str, str]
    gt_unseen_placements: Dict[str, str]
    annotator_notes: List[str]

    # evidence collected from user/oracle
    qa_history: List[QAItem]
    asked_questions: List[str]
    confirmed_actions: Dict[str, str]  # seen object -> receptacle

    # preference state
    open_preference_dimensions: List[str]
    preference_hypotheses: List[PreferenceHypothesis]

    # only seen objects are queried; unseen is evaluation-only
    unresolved_objects: List[str]

    # transient I/O
    current_question: Optional[str]
    current_answer: Optional[str]

    # final outputs
    predicted_placements_seen: Dict[str, str]
    predicted_placements_unseen: Dict[str, str]