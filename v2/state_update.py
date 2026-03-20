from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.agent_schema import (
        AgentState,
        ConfirmedPreference,
        IntentSource,
        QAItem,
        QuestionPattern,
    )
    from v2.data import DEFAULT_DATA_PATH, get_episode
    from v2.state_init import build_initial_state
except ModuleNotFoundError:
    from agent_schema import (
        AgentState,
        ConfirmedPreference,
        IntentSource,
        QAItem,
        QuestionPattern,
    )
    from data import DEFAULT_DATA_PATH, get_episode
    from state_init import build_initial_state


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
    update_type: Literal["direct_place", "exclude_receptacle", "general_rule", "no_update"]
    confirmed_action_receptacle: Optional[str] = Field(default=None)
    confirmed_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    excluded_receptacles: List[str] = Field(default_factory=list)
    confirmed_preference: Optional[PreferenceRecordModel] = None


class PreferenceElicitingInterpretation(BaseModel):
    update_type: Literal["confirmed_rule", "no_preference", "rule_with_exception", "no_update"]
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
- no_update:
  the answer does not support any reliable state update

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

Internal source:
action

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
        target: str,
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
  the answer says there is no special preference for this dimension
- rule_with_exception:
  the answer gives a stable rule plus one or more object-level exceptions
- no_update:
  the answer is too weak to support a reliable update

Rules:
- use only exact receptacle names from the provided receptacles
- use only exact seen object names in covered_objects and exception_actions
- if the answer gives a stable rule, put it in confirmed_preference with source = "elicited"
- confirmed_preference.covered_objects must be a subset of the current intent_covered_objects that is explicitly supported by the answer
- only use the full current intent_covered_objects when the answer clearly supports the whole set
- make the hypothesis concrete enough to be useful for placement, not just an abstract restatement of the dimension
- if there is no preference, add the target dimension or a short equivalent text to rejected_hypotheses
- rejected_hypotheses must be empty unless update_type = "no_preference"
- if there is an exception object with a clear placement, add it to exception_actions
- be conservative
""".strip()

        user_prompt = f"""
Question pattern:
preference_eliciting

Internal source:
scene_gap

Target dimension:
{target}

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

Current open_preference_dimensions:
{state["open_preference_dimensions"]}

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

    def update_open_preference_dimensions(
        self,
        *,
        state: AgentState,
    ) -> List[str]:
        system_prompt = """
You update open preference dimensions for a household rearrangement agent.

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

Rules:
- return short dimension names, not full hypotheses
- prefer dimensions that could still change placement decisions for unresolved objects
- do not repeat dimensions already settled by confirmed_preferences
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
            raw_dimensions = [item.strip() for item in result.dimensions if item.strip()]
        except Exception:
            raw_dimensions = []

        rejected = {_norm(item) for item in state["rejected_hypotheses"] if item.strip()}
        deduped: List[str] = []
        seen = set()
        for item in raw_dimensions:
            normalized = _norm(item)
            if normalized in seen or normalized in rejected:
                continue
            deduped.append(item)
            seen.add(normalized)

        state["open_preference_dimensions"] = deduped
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
    source: IntentSource,
    target: str,
    answer: str,
    question: Optional[str] = None,
    action_mode: Optional[str] = None,
) -> None:
    state["qa_history"].append(
        QAItem(
            question_pattern=question_pattern,
            source=source,
            target=target,
            action_mode=action_mode,
            question=question or state.get("current_question"),
            answer=answer,
        )
    )
    state["budget_used"] += 1
    state["current_pattern"] = question_pattern
    state["current_question"] = question or state.get("current_question")
    state["current_answer"] = answer


def _upsert_confirmed_preference(state: AgentState, preference: ConfirmedPreference) -> None:
    hypothesis_norm = _norm(preference.get("hypothesis", ""))
    if not hypothesis_norm:
        return
    for idx, existing in enumerate(state["confirmed_preferences"]):
        if _norm(existing.get("hypothesis", "")) == hypothesis_norm:
            state["confirmed_preferences"][idx] = preference
            return
    state["confirmed_preferences"].append(preference)


def _normalize_confirmed_preference(
    *,
    preference: PreferenceRecordModel,
    seen_objects: List[str],
    receptacles: List[str],
    default_source: Literal["elicited", "confirmed"],
    fallback_covered_objects: Optional[List[str]] = None,
) -> Optional[ConfirmedPreference]:
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

    return ConfirmedPreference(
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
        source="action",
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
    target: str,
    covered_objects: List[str],
    answer: str,
    interpretation: PreferenceElicitingInterpretation,
    question: Optional[str] = None,
) -> AgentState:
    _append_qa_history(
        state=state,
        question_pattern="preference_eliciting",
        source="scene_gap",
        target=target,
        answer=answer,
        question=question,
    )

    if interpretation.update_type in ("confirmed_rule", "no_preference", "rule_with_exception"):
        state["open_preference_dimensions"] = [
            dim for dim in state["open_preference_dimensions"] if _norm(dim) != _norm(target)
        ]

    if interpretation.update_type == "no_preference":
        rejected = interpretation.rejected_hypotheses or [target]
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
        source: IntentSource,
        target: str,
        answer: str,
        updater: StateUpdate,
        question: Optional[str] = None,
        action_mode: Optional[str] = None,
        covered_objects: Optional[List[str]] = None,
) -> AgentState:
    if question_pattern != "action_oriented":
        if question_pattern != "preference_eliciting":
            raise NotImplementedError("This minimal state_update version currently supports action_oriented and preference_eliciting only.")
        if source != "scene_gap":
            raise ValueError("preference_eliciting updates must use source='scene_gap'.")

        interpretation = updater.interpret_preference_eliciting_answer(
            state=state,
            target=target,
            covered_objects=covered_objects or [],
            answer=answer,
            question=question,
        )
        state = update_from_preference_eliciting_answer(
            state=state,
            target=target,
            covered_objects=covered_objects or [],
            answer=answer,
            interpretation=interpretation,
            question=question,
        )
        updater.update_open_preference_dimensions(state=state)
        return state
    if source != "action":
        raise ValueError("action_oriented updates must use source='action'.")

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
    updater.update_open_preference_dimensions(state=state)
    return state


def update_open_preference_dimensions(
    *,
    state: AgentState,
    updater: StateUpdate,
) -> AgentState:
    updater.update_open_preference_dimensions(state=state)
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


def _build_state_for_smoke(data_path: str, index: int) -> AgentState:
    episode = get_episode(Path(data_path), index)
    state = build_initial_state(
        episode=episode,
        strategy="parallel_exploration",
        budget_total=5,
    )
    state["open_preference_dimensions"] = [
        "fragility",
        "accessibility",
        "category_grouping",
    ]
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for minimal state update.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    args = parser.parse_args()

    episode = get_episode(Path(args.data), args.index)
    updater = StateUpdate(model=args.model, base_url=args.base_url, temperature=0.0)
    target = episode.seen_objects[0]
    question = f"Where should I place the {target}?"
    pref_target = "fragility"
    pref_question = "Do you have a preference for where fragile items should go?"

    direct_state = _build_state_for_smoke(args.data, args.index)
    direct_answer = f"Put it in the {episode.receptacles[0]}."
    direct_interpretation = updater.interpret_action_answer(
        state=direct_state,
        target=target,
        answer=direct_answer,
        question=question,
        action_mode="direct_grounding",
    )
    direct_state = update_from_action_answer(
        state=direct_state,
        target=target,
        answer=direct_answer,
        interpretation=direct_interpretation,
        question=question,
        action_mode="direct_grounding",
    )

    exclude_state = _build_state_for_smoke(args.data, args.index)
    exclude_answer = f"Not in the {episode.receptacles[0]}."
    exclude_interpretation = updater.interpret_action_answer(
        state=exclude_state,
        target=target,
        answer=exclude_answer,
        question=question,
        action_mode="boundary_probe",
    )
    exclude_state = update_from_action_answer(
        state=exclude_state,
        target=target,
        answer=exclude_answer,
        interpretation=exclude_interpretation,
        question=question,
        action_mode="boundary_probe",
    )

    rule_state = _build_state_for_smoke(args.data, args.index)
    rule_answer = f"{target}s usually go in the {episode.receptacles[0]}."
    rule_interpretation = updater.interpret_action_answer(
        state=rule_state,
        target=target,
        answer=rule_answer,
        question=question,
        action_mode="boundary_probe",
    )
    rule_state = update_from_action_answer(
        state=rule_state,
        target=target,
        answer=rule_answer,
        interpretation=rule_interpretation,
        question=question,
        action_mode="boundary_probe",
    )

    pref_rule_state = _build_state_for_smoke(args.data, args.index)
    pref_rule_answer = f"Fragile items usually go in the {episode.receptacles[0]}."
    pref_rule_interpretation = updater.interpret_preference_eliciting_answer(
        state=pref_rule_state,
        target=pref_target,
        covered_objects=episode.seen_objects[:2],
        answer=pref_rule_answer,
        question=pref_question,
    )
    pref_rule_state = update_from_preference_eliciting_answer(
        state=pref_rule_state,
        target=pref_target,
        covered_objects=episode.seen_objects[:2],
        answer=pref_rule_answer,
        interpretation=pref_rule_interpretation,
        question=pref_question,
    )

    no_pref_state = _build_state_for_smoke(args.data, args.index)
    no_pref_answer = "I do not have a special preference for that."
    no_pref_interpretation = updater.interpret_preference_eliciting_answer(
        state=no_pref_state,
        target=pref_target,
        covered_objects=episode.seen_objects[:2],
        answer=no_pref_answer,
        question=pref_question,
    )
    no_pref_state = update_from_preference_eliciting_answer(
        state=no_pref_state,
        target=pref_target,
        covered_objects=episode.seen_objects[:2],
        answer=no_pref_answer,
        interpretation=no_pref_interpretation,
        question=pref_question,
    )

    exception_state = _build_state_for_smoke(args.data, args.index)
    exception_object = episode.seen_objects[1] if len(episode.seen_objects) > 1 else episode.seen_objects[0]
    exception_receptacle = (
        episode.receptacles[1] if len(episode.receptacles) > 1 else episode.receptacles[0]
    )
    exception_answer = (
        f"Fragile items usually go in the {episode.receptacles[0]}, "
        f"but {exception_object} goes in the {exception_receptacle}."
    )
    exception_interpretation = updater.interpret_preference_eliciting_answer(
        state=exception_state,
        target=pref_target,
        covered_objects=episode.seen_objects[:3],
        answer=exception_answer,
        question=pref_question,
    )
    exception_state = update_from_preference_eliciting_answer(
        state=exception_state,
        target=pref_target,
        covered_objects=episode.seen_objects[:3],
        answer=exception_answer,
        interpretation=exception_interpretation,
        question=pref_question,
    )

    print(
        json.dumps(
            {
                "action_oriented": {
                    "direct_place": {
                        "interpretation": direct_interpretation.model_dump(),
                        "state": direct_state,
                    },
                    "exclude_receptacle": {
                        "interpretation": exclude_interpretation.model_dump(),
                        "state": exclude_state,
                    },
                    "general_rule": {
                        "interpretation": rule_interpretation.model_dump(),
                        "state": rule_state,
                    },
                },
                "preference_eliciting": {
                    "confirmed_rule": {
                        "interpretation": pref_rule_interpretation.model_dump(),
                        "state": pref_rule_state,
                    },
                    "no_preference": {
                        "interpretation": no_pref_interpretation.model_dump(),
                        "state": no_pref_state,
                    },
                    "rule_with_exception": {
                        "interpretation": exception_interpretation.model_dump(),
                        "state": exception_state,
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
