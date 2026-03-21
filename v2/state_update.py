from __future__ import annotations

import os
from typing import Any, Iterable, List, Literal, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.agent_schema import (
        AgentState,
        PreferenceRecord,
        QAItem,
        QuestionPattern,
    )
except ModuleNotFoundError:
    from agent_schema import (
        AgentState,
        PreferenceRecord,
        QAItem,
        QuestionPattern,
    )

QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class PreferenceRecordModel(BaseModel):
    hypothesis: str = Field(description="A concise confirmed preference rule.")
    source: Literal["elicited", "confirmed"] = "confirmed"
    covered_objects: List[str] = Field(default_factory=list)
    target_receptacle: Optional[str] = None
    exceptions: List[str] = Field(default_factory=list)


class ObjectPlacementModel(BaseModel):
    object_name: str
    receptacle: str


class ActionAnswerInterpretation(BaseModel):
    update_type: Literal["direct_place", "exclude_receptacle", "general_rule"]
    confirmed_action_receptacle: Optional[str] = Field(default=None)
    confirmed_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    excluded_receptacles: List[str] = Field(default_factory=list)
    confirmed_preference: Optional[PreferenceRecordModel] = None


class PreferenceElicitingInterpretation(BaseModel):
    update_type: Literal["confirmed_rule", "no_preference", "rule_with_exception"]
    confirmed_preference: Optional[PreferenceRecordModel] = None
    exception_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    rejected_hypotheses: List[str] = Field(default_factory=list)


class PreferenceSummaryInterpretation(BaseModel):
    update_types: List[Literal["confirmed_rule", "reject_summary", "rule_with_exception"]] = Field(
        default_factory=list
    )
    confirmed_preference: Optional[PreferenceRecordModel] = None
    exception_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    rejected_hypotheses: List[str] = Field(default_factory=list)


class OpenPreferenceDimensionsUpdate(BaseModel):
    dimensions: List[str] = Field(default_factory=list)


class StateUpdate:
    def __init__(
        self,
        model: str = QUESTION_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model: Any = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self.action_model = self.model.with_structured_output(ActionAnswerInterpretation)
        self.preference_eliciting_model = self.model.with_structured_output(PreferenceElicitingInterpretation)
        self.preference_summary_model = self.model.with_structured_output(PreferenceSummaryInterpretation)
        self.open_dimensions_update_model = self.model.with_structured_output(OpenPreferenceDimensionsUpdate)

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

Current excluded_receptacles:
{state["excluded_receptacles"]}

Current confirmed_preferences:
{state["confirmed_preferences"]}
""".strip()

        return self.action_model.invoke(
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
        question: Optional[str] = None,
    ) -> PreferenceElicitingInterpretation:
        system_prompt = """
You interpret a preference-eliciting answer for a household rearrangement agent.

Return exactly one update type:
- confirmed_rule:
  the answer gives a stable rule
- no_preference:
  the answer says there is no special preference for this hypothesis
- rule_with_exception:
  the answer gives a stable rule plus one or more object-level exceptions

Rules:
- use only exact receptacle names from the provided receptacles
- use only exact seen object names in covered_objects and exception_actions
- if the answer gives a stable rule, put it in confirmed_preference with source = "elicited"
- confirmed_preference.covered_objects must be a subset of the current intent_covered_objects that is explicitly supported by the answer
- only use the full current intent_covered_objects when the answer clearly supports the whole set
- make the hypothesis concrete enough to be useful for placement, not just an abstract restatement of the target hypothesis
- if there is no preference, add the target hypothesis or a short equivalent text to rejected_hypotheses
- rejected_hypotheses must be empty unless update_type = "no_preference"
- if there is an exception object with a clear placement, add it to exception_actions
- be conservative
""".strip()

        user_prompt = f"""
Question pattern:
preference_eliciting

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

Current open_preference_hypotheses:
{state["open_preference_hypotheses"]}

Current confirmed_actions:
{state["confirmed_actions"]}

Current confirmed_preferences:
{state["confirmed_preferences"]}

Current rejected_hypotheses:
{state["rejected_hypotheses"]}
""".strip()

        return self.preference_eliciting_model.invoke(
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

Return zero or more update types:
- confirmed_rule:
  the answer confirms the summary rule, possibly with refinement
- reject_summary:
  the answer rejects the current summary hypothesis
- rule_with_exception:
  the answer confirms a rule and also gives one or more object-level exceptions

You may return multiple update types when the answer does multiple things.
Example:
- reject the current summary but provide a better rule
- confirm a rule and also give object-level exceptions

Rules:
- use only exact receptacle names from the provided receptacles
- use only exact seen object names in covered_objects and exception_actions
- if the answer confirms or refines a stable rule, put it in confirmed_preference with source = "confirmed"
- confirmed_preference.covered_objects must be a subset of the current intent_covered_objects that is explicitly supported by the answer
- only use the full current intent_covered_objects when the answer clearly supports the whole set
- make the hypothesis concrete enough to be useful for placement
- if the summary is rejected, add the target hypothesis or a short equivalent text to rejected_hypotheses
- rejected_hypotheses should normally be non-empty when update_types includes "reject_summary"
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

Current preference_candidates:
{state["preference_candidates"]}

Current confirmed_preferences:
{state["confirmed_preferences"]}

Current rejected_hypotheses:
{state["rejected_hypotheses"]}
""".strip()

        return self.preference_summary_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    def update_from_preference_summary_answer(
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

        update_types = _dedupe_keep_order(list(interpretation.update_types))

        if "reject_summary" in update_types:
            _remove_preference_candidate(state, hypothesis)
            rejected = interpretation.rejected_hypotheses or [hypothesis]
            state["rejected_hypotheses"] = _dedupe_keep_order([*state["rejected_hypotheses"], *rejected])

        if interpretation.confirmed_preference is not None:
            _remove_preference_candidate(state, hypothesis)
            normalized = _normalize_confirmed_preference(
                preference=interpretation.confirmed_preference,
                seen_objects=state["seen_objects"],
                receptacles=state["receptacles"],
                default_source="confirmed",
                fallback_covered_objects=covered_objects,
            )
            if normalized is not None:
                if interpretation.exception_actions:
                    normalized["exceptions"] = _dedupe_keep_order(
                        [
                            *normalized.get("exceptions", []),
                            *[item.object_name for item in interpretation.exception_actions],
                        ]
                    )
                _upsert_confirmed_preference(state, normalized)

        _apply_confirmed_actions(state=state, placements=interpretation.exception_actions)

        return recompute_predictions(state)

    def update_open_preference_hypotheses(
        self,
        *,
        state: AgentState,
    ) -> List[str]:
        system_prompt = """
You update open preference hypotheses for a household rearrangement agent.

Your job:
- infer which high-level preference dimensions are still worth asking directly
- focus on unresolved objects and remaining uncertainty

Prioritize:
- unresolved_objects
- confirmed_preferences

You may also use:
- room
- receptacles
- seen_objects
- confirmed_actions
- rejected_hypotheses

Important:
- return high-level preference dimensions, not object names, not receptacle names, and not full placement rules

Rules:
- return short high-level hypothesis texts only
- prefer hypotheses that could still change placement decisions for unresolved objects
- do not repeat hypotheses already settled by confirmed_preferences
- do not repeat rejected_hypotheses
- return a small conservative list
""".strip()

        user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Unresolved objects:
{state["unresolved_objects"]}

Confirmed actions:
{state["confirmed_actions"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Rejected hypotheses:
{state["rejected_hypotheses"]}
""".strip()

        try:
            result = self.open_dimensions_update_model.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            raw_hypotheses = [item.strip() for item in result.dimensions if item.strip()]
        except Exception:
            raw_hypotheses = []

        rejected = {_norm(item) for item in state["rejected_hypotheses"] if item.strip()}
        seen_objects_norm = {_norm(item) for item in state["seen_objects"] if item.strip()}
        receptacles_norm = {_norm(item) for item in state["receptacles"] if item.strip()}
        deduped: List[str] = []
        seen = set()
        for item in raw_hypotheses:
            normalized = _norm(item)
            if not normalized:
                continue
            if normalized in seen or normalized in rejected:
                continue
            if normalized in seen_objects_norm or normalized in receptacles_norm:
                continue
            if any(obj in normalized for obj in seen_objects_norm):
                continue
            if any(rec in normalized for rec in receptacles_norm):
                continue
            deduped.append(item)
            seen.add(normalized)

        state["open_preference_hypotheses"] = deduped
        return deduped


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
    state["budget_used"] += 1


def _upsert_confirmed_preference(state: AgentState, preference: PreferenceRecord) -> None:
    hypothesis_norm = _norm(preference.get("hypothesis", ""))
    if not hypothesis_norm:
        return
    for idx, existing in enumerate(state["confirmed_preferences"]):
        if _norm(existing.get("hypothesis", "")) == hypothesis_norm:
            state["confirmed_preferences"][idx] = preference
            return
    state["confirmed_preferences"].append(preference)


def _remove_preference_candidate(state: AgentState, hypothesis: str) -> None:
    hypothesis_norm = _norm(hypothesis)
    if not hypothesis_norm:
        return
    state["preference_candidates"] = [
        item for item in state["preference_candidates"]
        if _norm(item.get("hypothesis", "")) != hypothesis_norm
    ]


def _normalize_confirmed_preference(
    *,
    preference: PreferenceRecordModel,
    seen_objects: List[str],
    receptacles: List[str],
    default_source: Literal["elicited", "confirmed"],
    fallback_covered_objects: Optional[List[str]] = None,
) -> Optional[PreferenceRecord]:
    hypothesis = preference.hypothesis.strip()
    if not hypothesis:
        return None

    allowed_objects = set(seen_objects)
    allowed_receptacles = set(receptacles)
    covered_objects = _dedupe_keep_order([obj for obj in preference.covered_objects if obj in allowed_objects])
    if fallback_covered_objects is not None:
        fallback = _dedupe_keep_order([obj for obj in fallback_covered_objects if obj in allowed_objects])
        if fallback:
            if covered_objects:
                fallback_set = set(fallback)
                covered_objects = [obj for obj in covered_objects if obj in fallback_set]
            else:
                covered_objects = fallback
    exceptions = _dedupe_keep_order([obj for obj in preference.exceptions if obj in allowed_objects])
    target_receptacle = preference.target_receptacle if preference.target_receptacle in allowed_receptacles else None

    return PreferenceRecord(
        hypothesis=hypothesis,
        source=preference.source if preference.source in ("elicited", "confirmed") else default_source,
        covered_objects=covered_objects,
        target_receptacle=target_receptacle,
        exceptions=exceptions,
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
            state["confirmed_actions"][item.object_name] = item.receptacle


def update_from_action_answer(
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
            state["confirmed_actions"][target] = receptacle

    elif interpretation.update_type == "exclude_receptacle":
        excluded = [r for r in interpretation.excluded_receptacles if r in allowed_receptacles]
        if excluded:
            existing = state["excluded_receptacles"].get(target, [])
            state["excluded_receptacles"][target] = _dedupe_keep_order([*existing, *excluded])
            remaining = [r for r in state["receptacles"] if r not in state["excluded_receptacles"][target]]
            if len(remaining) == 1:
                state["confirmed_actions"][target] = remaining[0]

    elif interpretation.update_type == "general_rule" and interpretation.confirmed_preference is not None:
        normalized = _normalize_confirmed_preference(
            preference=interpretation.confirmed_preference,
            seen_objects=state["seen_objects"],
            receptacles=state["receptacles"],
            default_source="confirmed",
        )
        if normalized is not None:
            _upsert_confirmed_preference(state, normalized)

    _apply_confirmed_actions(state=state, placements=interpretation.confirmed_actions)

    return recompute_predictions(state)


def update_from_preference_eliciting_answer(
    *,
    state: AgentState,
    hypothesis: str,
    covered_objects: List[str],
    answer: str,
    interpretation: PreferenceElicitingInterpretation,
    question: Optional[str] = None,
) -> AgentState:
    _append_qa_history(
        state=state,
        question_pattern="preference_eliciting",
        target=hypothesis,
        answer=answer,
        question=question,
    )

    if interpretation.update_type in ("confirmed_rule", "no_preference", "rule_with_exception"):
        state["open_preference_hypotheses"] = [
            item for item in state["open_preference_hypotheses"] if _norm(item) != _norm(hypothesis)
        ]

    if interpretation.update_type == "no_preference":
        rejected = interpretation.rejected_hypotheses or [hypothesis]
        state["rejected_hypotheses"] = _dedupe_keep_order([*state["rejected_hypotheses"], *rejected])
        return recompute_predictions(state)

    if interpretation.confirmed_preference is not None:
        normalized = _normalize_confirmed_preference(
            preference=interpretation.confirmed_preference,
            seen_objects=state["seen_objects"],
            receptacles=state["receptacles"],
            default_source="elicited",
            fallback_covered_objects=covered_objects,
        )
        if normalized is not None:
            if interpretation.exception_actions:
                normalized["exceptions"] = _dedupe_keep_order(
                    [
                        *normalized.get("exceptions", []),
                        *[item.object_name for item in interpretation.exception_actions],
                    ]
                )
            _upsert_confirmed_preference(state, normalized)

    _apply_confirmed_actions(state=state, placements=interpretation.exception_actions)

    if interpretation.update_type == "no_preference" and interpretation.rejected_hypotheses:
        state["rejected_hypotheses"] = _dedupe_keep_order(
            [*state["rejected_hypotheses"], *interpretation.rejected_hypotheses]
        )

    return recompute_predictions(state)


def update_state_with_answer(
    *,
    state: AgentState,
    question_pattern: QuestionPattern,
    target: str,
    answer: str,
    updater: StateUpdate,
    question: Optional[str] = None,
    action_mode: Optional[str] = None,
    covered_objects: Optional[List[str]] = None,
) -> AgentState:
    if question_pattern == "action_oriented":
        interpretation = updater.interpret_action_answer(
            state=state,
            target=target,
            answer=answer,
            question=question,
            action_mode=action_mode,
        )
        state = update_from_action_answer(
            state=state,
            target=target,
            answer=answer,
            interpretation=interpretation,
            question=question,
            action_mode=action_mode,
        )
        return state

    if question_pattern == "preference_eliciting":
        print(f"[eliciting] interpreting answer for hypothesis: {target}")
        interpretation = updater.interpret_preference_eliciting_answer(
            state=state,
            hypothesis=target,
            covered_objects=covered_objects or [],
            answer=answer,
            question=question,
        )
        print(f"[eliciting] interpretation finished: {interpretation.update_type}")
        state = update_from_preference_eliciting_answer(
            state=state,
            hypothesis=target,
            covered_objects=covered_objects or [],
            answer=answer,
            interpretation=interpretation,
            question=question,
        )
        print("[eliciting] state write finished; refreshing open preference hypotheses...")
        # Only elicitation updates should refresh open elicitation hypotheses.
        updater.update_open_preference_hypotheses(state=state)
        print("[eliciting] open preference hypotheses refresh finished")
        return state

    if question_pattern == "preference_summary":
        interpretation = updater.interpret_preference_summary_answer(
            state=state,
            hypothesis=target,
            covered_objects=covered_objects or [],
            answer=answer,
            question=question,
        )
        state = updater.update_from_preference_summary_answer(
            state=state,
            hypothesis=target,
            covered_objects=covered_objects or [],
            answer=answer,
            interpretation=interpretation,
            question=question,
        )
        return state

    raise NotImplementedError(
        "This minimal state_update version currently supports action_oriented, preference_eliciting, and preference_summary only."
    )


def update_open_preference_hypotheses(
    *,
    state: AgentState,
    updater: StateUpdate,
) -> AgentState:
    updater.update_open_preference_hypotheses(state=state)
    return state


def recompute_predictions(state: AgentState) -> AgentState:
    predicted_seen = dict(state["confirmed_actions"])
    # At this stage we only ground placements for seen objects.
    # Keep unseen predictions empty until a later evaluation phase.
    predicted_unseen = {}

    for preference in state["confirmed_preferences"]:
        receptacle = preference.get("target_receptacle")
        if not receptacle:
            continue

        exceptions = set(preference.get("exceptions", []))
        for obj in preference.get("covered_objects", []):
            if obj in state["seen_objects"] and obj not in predicted_seen and obj not in exceptions:
                predicted_seen[obj] = receptacle

    for obj, excluded in state["excluded_receptacles"].items():
        if predicted_seen.get(obj) in excluded:
            predicted_seen.pop(obj, None)

    state["predicted_placements_seen"] = predicted_seen
    state["predicted_placements_unseen"] = predicted_unseen
    state["unresolved_objects"] = [
        obj for obj in state["seen_objects"] if obj not in state["confirmed_actions"] and obj not in predicted_seen
    ]
    return state
