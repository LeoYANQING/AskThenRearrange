from __future__ import annotations

import os
from typing import Any, Iterable, List, Literal, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.agent_schema import (
        AgentState,
        LearnedPreference,
        QAItem,
        QuestionPattern,
    )
except ModuleNotFoundError:
    from agent_schema import (
        AgentState,
        LearnedPreference,
        QAItem,
        QuestionPattern,
    )

QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class LearnedPreferenceModel(BaseModel):
    hypothesis: str = Field(description="A concise confirmed preference rule.")
    covered_objects: List[str] = Field(default_factory=list)


class ObjectPlacementModel(BaseModel):
    object_name: str
    receptacle: str


class ActionAnswerInterpretation(BaseModel):
    update_type: Literal["direct_place", "exclude_receptacle", "general_rule"]
    confirmed_action_receptacle: Optional[str] = Field(default=None)
    confirmed_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    excluded_receptacles: List[str] = Field(default_factory=list)
    confirmed_preference: Optional[LearnedPreferenceModel] = None


class PreferenceElicitingStateUpdate(BaseModel):
    confirmed_preference: Optional[LearnedPreferenceModel] = None
    confirmed_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    negative_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    negative_preference: Optional[str] = None


class PreferenceSummaryInterpretation(BaseModel):
    update_type: Literal["confirmed_rule", "reject_summary", "rule_with_exception"]
    confirmed_preference: Optional[LearnedPreferenceModel] = None
    exception_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    negative_preferences: List[str] = Field(default_factory=list)


class StateUpdate:
    def __init__(
        self,
        model: str = QUESTION_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.model: Any = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
        )
        self.action_model = self.model.with_structured_output(ActionAnswerInterpretation)
        self.preference_eliciting_model = self.model.with_structured_output(PreferenceElicitingStateUpdate)
        self.preference_summary_model = self.model.with_structured_output(PreferenceSummaryInterpretation)

    def interpret_action_answer(
        self,
        *,
        state: AgentState,
        target: str,
        answer: str,
        question: Optional[str] = None,
        action_mode: Optional[str] = None,
    ) -> ActionAnswerInterpretation:
        system_prompt = """
You interpret an action-oriented answer for a household rearrangement agent.

Return exactly one update type:
- direct_place:
  the answer gives a direct receptacle for the target object
- exclude_receptacle:
  the answer rules out one or more receptacles for the target object
- general_rule:
  the answer upgrades from object placement to a general rule

Rules:
- use only exact receptacle names from the provided receptacles
- use only exact seen object names in confirmed_actions, covered_objects, and exceptions
- be conservative
- if the answer is a general rule, put it in confirmed_preference with source = "confirmed"
- if the answer is only a direct placement, do not invent a rule
- if the answer gives any object-level placement that is explicitly supported, add it to confirmed_actions
- confirmed_actions may include the target object and any clearly stated object-level exceptions to a general rule
""".strip()

        user_prompt = f"""
Question pattern:
action_oriented

Target object:
{target}

Action mode:
{action_mode}

Question:
{question}

Answer:
{answer}

Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Current confirmed_actions:
{state["confirmed_actions"]}

Current confirmed_preferences:
{state["confirmed_preferences"]}
""".strip()

        return self.action_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    def interpret_preference_summary_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: List[str],
        answer: str,
        question: Optional[str] = None,
    ) -> PreferenceSummaryInterpretation:
        system_prompt = """
You interpret a preference-summary answer for a household rearrangement agent.

Return exactly one update type:
- confirmed_rule:
  the answer confirms or refines a summary rule that is useful for future placement decisions
- reject_summary:
  the answer rejects the current summary hypothesis
- rule_with_exception:
  the answer confirms a useful rule and also gives one or more object-level exceptions

Rules:
- use only exact receptacle names from the provided receptacles
- use only exact seen object names in covered_objects and exception_actions
- if the answer confirms or refines a stable rule, put it in confirmed_preference with source = "confirmed"
- confirmed_preference.covered_objects must be a subset of the current intent_covered_objects that is explicitly supported by the answer
- only use the full current intent_covered_objects when the answer clearly supports the whole set
- make the confirmed_preference concrete enough to guide future placements, not just restate the summary vaguely
- if the summary is rejected, add the target hypothesis or a short equivalent text to negative_preferences
- negative_preferences should normally be non-empty when update_type = "reject_summary"
- if the user rejects the current summary but provides a better stable rule, prefer returning confirmed_rule or rule_with_exception instead of reject_summary
- if there is an exception object with a clear placement, add it to exception_actions
- be conservative
""".strip()

        user_prompt = f"""
Question pattern:
preference_summary

Target hypothesis:
{hypothesis}

Current intent covered_objects:
{covered_objects}

Question:
{question}

Answer:
{answer}

Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Current confirmed_actions:
{state["confirmed_actions"]}

Current confirmed_preferences:
{state["confirmed_preferences"]}

Current negative_preferences:
{state["negative_preferences"]}
""".strip()

        return self.preference_summary_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    def interpret_preference_eliciting_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: List[str],
        answer: str,
    ) -> PreferenceElicitingStateUpdate:
        system_prompt = """
Interpret one preference-eliciting answer for a household rearrangement agent.

Return only the minimal update for the current AgentState.

Allowed outputs:
- confirmed_preference: a short family-level preference the answer supports
- negative_preference: a short rejected hypothesis
- confirmed_actions: exact seen object -> receptacle facts explicitly stated
- negative_actions: exact seen object -> receptacle facts explicitly ruled out

Rules:
- be conservative
- use only exact seen object names
- use only exact receptacle names
- confirmed_preference should be a short grouping preference, not a placement sentence
- if the answer rejects the current hypothesis, fill negative_preference
- if the answer confirms the current hypothesis, prefer keeping it close to the original wording
""".strip()

        user_prompt = f"""
Hypothesis:
{hypothesis}

Covered objects:
{covered_objects}

Answer:
{answer}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}
""".strip()

        return self.preference_eliciting_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    def apply_preference_summary_interpretation(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: List[str],
        answer: str,
        interpretation: PreferenceSummaryInterpretation,
        question: Optional[str] = None,
    ) -> AgentState:
        _append_qa_history(
            state=state,
            question_pattern="preference_summary",
            target=hypothesis,
            answer=answer,
            question=question,
        )

        if interpretation.update_type == "reject_summary":
            rejected = interpretation.negative_preferences or [hypothesis]
            for item in rejected:
                _upsert_negative_preference(
                    state,
                    hypothesis=item,
                    covered_objects=covered_objects,
                )
            _apply_confirmed_actions(state=state, placements=interpretation.exception_actions)
            _remove_negative_actions_for_confirmed(state=state, placements=interpretation.exception_actions)
            return recompute_online_placements(state)

        normalized = None
        if interpretation.confirmed_preference is not None:
            normalized = _normalize_confirmed_preference(
                preference=interpretation.confirmed_preference,
                seen_objects=state["seen_objects"],
                fallback_covered_objects=covered_objects,
            )

        if interpretation.update_type == "confirmed_rule":
            if normalized is not None:
                _upsert_confirmed_preference(state, normalized)
            return recompute_online_placements(state)

        if interpretation.update_type == "rule_with_exception":
            if normalized is not None:
                _upsert_confirmed_preference(state, normalized)
            _apply_confirmed_actions(state=state, placements=interpretation.exception_actions)
            _remove_negative_actions_for_confirmed(state=state, placements=interpretation.exception_actions)
            return recompute_online_placements(state)

        raise ValueError(f"Unsupported preference summary update_type: {interpretation.update_type}")

    def apply_action_interpretation(
        self,
        *,
        state: AgentState,
        target: str,
        answer: str,
        interpretation: ActionAnswerInterpretation,
        question: Optional[str] = None,
        action_mode: Optional[str] = None,
    ) -> AgentState:
        _append_qa_history(
            state=state,
            question_pattern="action_oriented",
            target=target,
            answer=answer,
            question=question,
            action_mode=action_mode,
        )

        allowed_receptacles = set(state["receptacles"])
        if interpretation.update_type == "direct_place":
            receptacle = interpretation.confirmed_action_receptacle
            if receptacle in allowed_receptacles:
                _upsert_confirmed_action(
                    state,
                    object_name=target,
                    receptacle=receptacle,
                )
                _remove_negative_action(state=state, object_name=target, receptacle=receptacle)

        elif interpretation.update_type == "general_rule" and interpretation.confirmed_preference is not None:
            normalized = _normalize_confirmed_preference(
                preference=interpretation.confirmed_preference,
                seen_objects=state["seen_objects"],
            )
            if normalized is not None:
                _upsert_confirmed_preference(state, normalized)

        _apply_confirmed_actions(state=state, placements=interpretation.confirmed_actions)
        if interpretation.update_type == "exclude_receptacle":
            _apply_negative_action_receptacles(
                state=state,
                target=target,
                receptacles=interpretation.excluded_receptacles,
            )

        return recompute_online_placements(state)

    def update_state_from_action_answer(
        self,
        *,
        state: AgentState,
        target: str,
        answer: str,
        question: Optional[str] = None,
        action_mode: Optional[str] = None,
    ) -> AgentState:
        interpretation = self.interpret_action_answer(
            state=state,
            target=target,
            answer=answer,
            question=question,
            action_mode=action_mode,
        )
        return self.apply_action_interpretation(
            state=state,
            target=target,
            answer=answer,
            interpretation=interpretation,
            question=question,
            action_mode=action_mode,
        )

    def update_state_from_preference_eliciting_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: Optional[List[str]],
        answer: str,
        question: Optional[str] = None,
    ) -> AgentState:
        update = self.interpret_preference_eliciting_answer(
            state=state,
            hypothesis=hypothesis,
            covered_objects=covered_objects or [],
            answer=answer,
        )
        _append_qa_history(
            state=state,
            question_pattern="preference_eliciting",
            target=hypothesis,
            answer=answer,
            question=question,
        )
        if update.negative_preference:
            _upsert_negative_preference(
                state,
                hypothesis=update.negative_preference,
                covered_objects=covered_objects or [],
            )
        if update.confirmed_preference is not None:
            normalized = _normalize_confirmed_preference(
                preference=update.confirmed_preference,
                seen_objects=state["seen_objects"],
                fallback_covered_objects=covered_objects or [],
            )
            if normalized is not None:
                _upsert_confirmed_preference(state, normalized)
        _apply_confirmed_actions(state=state, placements=update.confirmed_actions)
        _remove_negative_actions_for_confirmed(state=state, placements=update.confirmed_actions)
        _apply_negative_actions(state=state, placements=update.negative_actions)
        return recompute_online_placements(state)

    def update_state_from_preference_summary_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: Optional[List[str]],
        answer: str,
        question: Optional[str] = None,
    ) -> AgentState:
        interpretation = self.interpret_preference_summary_answer(
            state=state,
            hypothesis=hypothesis,
            covered_objects=covered_objects or [],
            answer=answer,
            question=question,
        )
        return self.apply_preference_summary_interpretation(
            state=state,
            hypothesis=hypothesis,
            covered_objects=covered_objects or [],
            answer=answer,
            interpretation=interpretation,
            question=question,
        )

def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _apply_negative_action_receptacles(
    *,
    state: AgentState,
    target: str,
    receptacles: List[str],
) -> None:
    allowed_objects = set(state["seen_objects"])
    allowed_receptacles = set(state["receptacles"])
    if target not in allowed_objects:
        return
    existing = {(item["object_name"], item["receptacle"]) for item in state["negative_actions"]}
    for receptacle in receptacles:
        if receptacle not in allowed_receptacles:
            continue
        key = (target, receptacle)
        if key in existing:
            continue
        state["negative_actions"].append({"object_name": target, "receptacle": receptacle})
        existing.add(key)


def _remove_negative_action(state: AgentState, *, object_name: str, receptacle: str) -> None:
    state["negative_actions"] = [
        item for item in state["negative_actions"]
        if not (item["object_name"] == object_name and item["receptacle"] == receptacle)
    ]


def _remove_negative_actions_for_confirmed(
    *,
    state: AgentState,
    placements: Iterable[ObjectPlacementModel],
) -> None:
    for item in placements:
        _remove_negative_action(state, object_name=item.object_name, receptacle=item.receptacle)


def _apply_negative_actions(
    *,
    state: AgentState,
    placements: Iterable[ObjectPlacementModel],
) -> None:
    allowed_objects = set(state["seen_objects"])
    allowed_receptacles = set(state["receptacles"])
    existing = {(item["object_name"], item["receptacle"]) for item in state["negative_actions"]}
    for item in placements:
        if item.object_name not in allowed_objects or item.receptacle not in allowed_receptacles:
            continue
        key = (item.object_name, item.receptacle)
        if key in existing:
            continue
        state["negative_actions"].append(
            {"object_name": item.object_name, "receptacle": item.receptacle}
        )
        existing.add(key)


def _append_qa_history(
    *,
    state: AgentState,
    question_pattern: QuestionPattern,
    target: str,
    answer: str,
    question: Optional[str] = None,
    action_mode: Optional[str] = None,
) -> None:
    state["qa_history"].append(
        QAItem(
            question_pattern=question_pattern,
            target=target,
            action_mode=action_mode,
            question=question or "",
            answer=answer,
        )
    )


def _confirmed_action_map(state: AgentState) -> dict[str, str]:
    return {
        item["object_name"]: item["receptacle"]
        for item in state["confirmed_actions"]
    }


def _confirmed_action_objects(state: AgentState) -> set[str]:
    return {item["object_name"] for item in state["confirmed_actions"]}


def _upsert_confirmed_action(
    state: AgentState,
    *,
    object_name: str,
    receptacle: str,
) -> None:
    for idx, existing in enumerate(state["confirmed_actions"]):
        if existing["object_name"] == object_name:
            state["confirmed_actions"][idx] = {
                "object_name": object_name,
                "receptacle": receptacle,
            }
            return
    state["confirmed_actions"].append(
        {"object_name": object_name, "receptacle": receptacle}
    )


def _upsert_confirmed_preference(state: AgentState, preference: LearnedPreference) -> None:
    hypothesis_norm = _norm(preference.get("hypothesis", ""))
    if not hypothesis_norm:
        return
    for idx, existing in enumerate(state["confirmed_preferences"]):
        if _norm(existing.get("hypothesis", "")) == hypothesis_norm:
            state["confirmed_preferences"][idx] = preference
            return
    state["confirmed_preferences"].append(preference)


def _upsert_negative_preference(
    state: AgentState,
    *,
    hypothesis: str,
    covered_objects: List[str],
) -> None:
    hypothesis_norm = _norm(hypothesis)
    if not hypothesis_norm:
        return
    allowed_objects = set(state["seen_objects"])
    normalized = LearnedPreference(
        hypothesis=hypothesis.strip(),
        covered_objects=_dedupe_keep_order(
            [obj for obj in covered_objects if obj in allowed_objects]
        ),
    )
    for idx, existing in enumerate(state["negative_preferences"]):
        if _norm(existing.get("hypothesis", "")) == hypothesis_norm:
            state["negative_preferences"][idx] = normalized
            return
    state["negative_preferences"].append(normalized)


def _normalize_confirmed_preference(
    *,
    preference: LearnedPreferenceModel,
    seen_objects: List[str],
    fallback_covered_objects: Optional[List[str]] = None,
) -> Optional[LearnedPreference]:
    hypothesis = preference.hypothesis.strip()
    if not hypothesis:
        return None

    allowed_objects = set(seen_objects)
    covered_objects = _dedupe_keep_order([obj for obj in preference.covered_objects if obj in allowed_objects])
    if fallback_covered_objects is not None:
        fallback = _dedupe_keep_order([obj for obj in fallback_covered_objects if obj in allowed_objects])
        if fallback:
            if covered_objects:
                fallback_set = set(fallback)
                covered_objects = [obj for obj in covered_objects if obj in fallback_set]
            else:
                covered_objects = fallback

    return LearnedPreference(
        hypothesis=hypothesis,
        covered_objects=covered_objects,
    )


def _apply_confirmed_actions(
    *,
    state: AgentState,
    placements: Iterable[ObjectPlacementModel],
) -> None:
    allowed_objects = set(state["seen_objects"])
    allowed_receptacles = set(state["receptacles"])
    for item in placements:
        if item.object_name in allowed_objects and item.receptacle in allowed_receptacles:
            _upsert_confirmed_action(
                state,
                object_name=item.object_name,
                receptacle=item.receptacle,
            )




def recompute_online_placements(state: AgentState) -> AgentState:
    confirmed_objects = {item["object_name"] for item in state["confirmed_actions"]}
    state["unresolved_objects"] = [
        obj for obj in state["seen_objects"] if obj not in confirmed_objects
    ]
    return state
