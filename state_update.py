from __future__ import annotations

import os
from typing import Any, Iterable, List, Literal, Optional

from llm_factory import create_chat_model, DEFAULT_MODEL, DEFAULT_BASE_URL
from pydantic import BaseModel, Field

from agent_schema import (
    AgentState,
    LearnedPreference,
    QAItem,
    QuestionPattern,
)

# Model config now in llm_factory.py
# Base URL config now in llm_factory.py


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
    category_rule: str = Field(
        default="",
        description=(
            "A short generalizable category-level organizing rule the answer supports. "
            "Fill this when the answer reveals a habit or principle like 'I always keep X in Y' or 'Y category belongs in Z'. "
            "Leave empty string only if the answer contains no organizing rule at all."
        ),
    )
    category_rule_covered_objects: List[str] = Field(
        default_factory=list,
        description="Exact seen object names covered by the category_rule. Must be a subset of seen_objects.",
    )
    category_rule_receptacle: str = Field(
        default="",
        description=(
            "Exact receptacle name from the receptacles list if the category_rule resolves to one specific place. "
            "Leave empty if the rule is ambiguous, conditional, or not tied to a single receptacle."
        ),
    )
    confirmed_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    negative_actions: List[ObjectPlacementModel] = Field(default_factory=list)
    negative_preference: Optional[str] = None


class PreferenceInductionInterpretation(BaseModel):
    update_type: Literal["confirmed_rule", "reject_induction", "rule_with_exception"]
    confirmed_hypothesis: str = Field(
        default="",
        description=(
            "The confirmed or refined preference rule as a concise sentence. "
            "Fill when update_type is 'confirmed_rule' or 'rule_with_exception'. "
            "Leave empty for 'reject_induction'."
        ),
    )
    confirmed_covered_objects: List[str] = Field(
        default_factory=list,
        description=(
            "Exact seen object names explicitly supported by the confirmed rule. "
            "Must be a subset of intent_covered_objects. "
            "Leave empty to inherit all intent_covered_objects."
        ),
    )
    confirmed_receptacle: Optional[str] = Field(
        default=None,
        description=(
            "Exact receptacle name from the provided receptacles list if the confirmed rule resolves "
            "to one specific place. Fill when update_type is 'confirmed_rule' or 'rule_with_exception' "
            "and the answer clearly points to a single receptacle. Leave null if ambiguous."
        ),
    )
    exception_object_name: str = Field(
        default="",
        description="Exact seen object name that is an exception to the rule. Fill only for 'rule_with_exception'.",
    )
    exception_receptacle: str = Field(
        default="",
        description="Exact receptacle name where the exception object should go. Fill only for 'rule_with_exception'.",
    )
    negative_preferences: List[str] = Field(
        default_factory=list,
        description="Hypothesis text(s) to reject. Fill when update_type is 'reject_induction'.",
    )


def _invoke_with_retry(model: Any, messages: list, retries: int = 3) -> Any:
    """Invoke a structured output model with retry on parse failures."""
    last_exc = None
    for attempt in range(retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                continue
    raise last_exc  # type: ignore[misc]


class StateUpdate:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.model: Any = create_chat_model(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
            timeout=120,
        )
        self.action_model = self.model.with_structured_output(ActionAnswerInterpretation)
        self.preference_eliciting_model = self.model.with_structured_output(PreferenceElicitingStateUpdate)
        self.preference_induction_model = self.model.with_structured_output(PreferenceInductionInterpretation)

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

        return _invoke_with_retry(self.action_model, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

    def interpret_preference_induction_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: List[str],
        answer: str,
        question: Optional[str] = None,
    ) -> PreferenceInductionInterpretation:
        system_prompt = """
You interpret a preference-induction answer for a household rearrangement agent.

The agent proposed a hypothesis about how a group of objects should be organized and asked the user to confirm or refine it.

Return exactly one update type:
- confirmed_rule: the answer confirms or refines the hypothesis as a stable organizing rule
- reject_induction: the answer rejects the hypothesis as incorrect or inapplicable
- rule_with_exception: the answer confirms the rule but mentions one specific object that is an exception

Field instructions:
- confirmed_hypothesis: fill when update_type is confirmed_rule or rule_with_exception
  - write the confirmed (or refined) rule as one concise sentence
  - if the user refines the hypothesis, write the refined version
  - example: "everyday kitchenware should be stored in the kitchen cabinet"
  - leave empty string for reject_induction
- confirmed_covered_objects: exact seen object names from intent_covered_objects that the confirmed rule applies to
  - leave empty to inherit all intent_covered_objects
- confirmed_receptacle: exact receptacle name from the provided receptacles list if the confirmed rule points to one place
  - fill when the answer clearly identifies a single destination (e.g. "Yes, they all go on the bookshelf")
  - must be an exact name from the receptacles list
  - leave null if the answer is ambiguous or covers multiple locations
- exception_object_name: fill only for rule_with_exception — one exact seen object name that is an exception
- exception_receptacle: fill only for rule_with_exception — exact receptacle for the exception object
- negative_preferences: fill when update_type is reject_induction — include the rejected hypothesis or a short summary of what was rejected

Rules:
- use only exact receptacle names from the provided receptacles list
- use only exact seen object names from the seen_objects list
- if the user provides a corrected rule instead of a flat rejection, prefer confirmed_rule over reject_induction
- do not invent placements not stated in the answer
- be conservative
""".strip()

        user_prompt = f"""
Question pattern:
preference_induction

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

        return _invoke_with_retry(self.preference_induction_model, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

    def interpret_preference_eliciting_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: List[str],
        answer: str,
        question: Optional[str] = None,
    ) -> PreferenceElicitingStateUpdate:
        system_prompt = """
Interpret one preference-eliciting answer for a household rearrangement agent.

The question was about the user's organizing HABIT or PRINCIPLE for a category. The answer may describe
how the user organizes, why items go to a certain spot, or confirm/deny a suggested location.

Output fields:
- category_rule: a short generalizable organizing rule extracted from the answer.
  Fill this whenever the answer reveals a habit or principle, even if phrased as "I usually keep X in Y"
  or "I always put those near Z" or "they tend to stay in one spot".
  Leave empty ONLY if the answer contains no organizing rule at all.
- category_rule_covered_objects: seen objects this rule applies to (subset of seen_objects).
- category_rule_receptacle: exact receptacle name if the rule resolves to ONE specific place.
  Leave empty if the answer is ambiguous ("somewhere in the living room"), conditional, or multi-location.
- confirmed_actions: object -> receptacle mappings EXPLICITLY stated for a SINGLE named object with ONE
  unambiguous receptacle. If the answer gives "X or Y" options, do NOT add to confirmed_actions.
- negative_actions: only when the answer EXPLICITLY says an object should NOT go to a specific receptacle.
- negative_preference: only if the user explicitly rejects the hypothesis.

Rules:
- use only exact seen object names from seen_objects
- use only exact receptacle names from receptacles
- category_rule should be a category-level rule, not a per-object placement sentence
  - good: "writing and memory items belong on the reading shelf"
  - good: "I always keep cleaning supplies under the sink"
  - good: "electronic accessories tend to stay in the media console"
  - bad: "the salt shaker goes on the countertop"
- category_rule_receptacle must be an exact match from the provided receptacles list
- category_rule and confirmed_actions are not mutually exclusive — fill both when applicable
- DO NOT add to negative_actions because an object was not mentioned
""".strip()

        user_prompt = f"""
Hypothesis:
{hypothesis}

Covered objects:
{covered_objects}

Question:
{question or ""}

Answer:
{answer}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Unresolved objects (still need placement):
{state["unresolved_objects"]}

Current confirmed_actions:
{state["confirmed_actions"]}
""".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return _invoke_with_retry(self.preference_eliciting_model, messages)
        except Exception:
            # Fallback: extract a category rule from the answer text directly
            receptacle = ""
            for r in state["receptacles"]:
                if r.lower() in answer.lower():
                    receptacle = r
                    break
            return PreferenceElicitingStateUpdate(
                category_rule=answer.strip() if receptacle else "",
                category_rule_covered_objects=[
                    obj for obj in covered_objects if obj in state["seen_objects"]
                ],
                category_rule_receptacle=receptacle,
            )

    def apply_preference_induction_interpretation(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: List[str],
        answer: str,
        interpretation: PreferenceInductionInterpretation,
        question: Optional[str] = None,
    ) -> AgentState:
        _append_qa_history(
            state=state,
            question_pattern="preference_induction",
            target=hypothesis,
            answer=answer,
            question=question,
        )

        if interpretation.update_type == "reject_induction":
            rejected = interpretation.negative_preferences or [hypothesis]
            for item in rejected:
                _upsert_negative_preference(
                    state,
                    hypothesis=item,
                    covered_objects=covered_objects,
                )
            _apply_single_exception(state=state, interpretation=interpretation)
            return recompute_online_placements(state)

        # Build normalized preference from flat fields
        normalized = None
        if interpretation.confirmed_hypothesis.strip():
            normalized = _normalize_confirmed_preference(
                preference=LearnedPreferenceModel(
                    hypothesis=interpretation.confirmed_hypothesis.strip(),
                    covered_objects=interpretation.confirmed_covered_objects,
                ),
                seen_objects=state["seen_objects"],
                fallback_covered_objects=covered_objects,
            )

        if interpretation.update_type == "confirmed_rule":
            if normalized is not None:
                resolved = _fuzzy_match_receptacle(
                    interpretation.confirmed_receptacle or "", state["receptacles"]
                ) or _auto_resolve_from_hypothesis(
                    state=state,
                    hypothesis_text=interpretation.confirmed_hypothesis,
                    covered_objects=normalized.get("covered_objects", []),
                )
                if resolved:
                    normalized["receptacle"] = resolved
                    unresolved_set = set(state["unresolved_objects"])
                    for obj in normalized.get("covered_objects", []):
                        if obj in unresolved_set:
                            _upsert_confirmed_action(state, object_name=obj, receptacle=resolved)
                _upsert_confirmed_preference(state, normalized)
            return recompute_online_placements(state)

        if interpretation.update_type == "rule_with_exception":
            if normalized is not None:
                resolved = _fuzzy_match_receptacle(
                    interpretation.confirmed_receptacle or "", state["receptacles"]
                ) or _auto_resolve_from_hypothesis(
                    state=state,
                    hypothesis_text=interpretation.confirmed_hypothesis,
                    covered_objects=normalized.get("covered_objects", []),
                )
                if resolved:
                    normalized["receptacle"] = resolved
                    unresolved_set = set(state["unresolved_objects"])
                    for obj in normalized.get("covered_objects", []):
                        if obj in unresolved_set:
                            _upsert_confirmed_action(state, object_name=obj, receptacle=resolved)
                _upsert_confirmed_preference(state, normalized)
            _apply_single_exception(state=state, interpretation=interpretation)
            return recompute_online_placements(state)

        raise ValueError(f"Unsupported preference induction update_type: {interpretation.update_type}")

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
                resolved = _auto_resolve_from_hypothesis(
                    state=state,
                    hypothesis_text=interpretation.confirmed_preference.hypothesis,
                    covered_objects=normalized.get("covered_objects", []),
                )
                if resolved:
                    normalized["receptacle"] = resolved
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
        oracle_receptacle: Optional[str] = None,
    ) -> AgentState:
        update = self.interpret_preference_eliciting_answer(
            state=state,
            hypothesis=hypothesis,
            covered_objects=covered_objects or [],
            answer=answer,
            question=question,
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
        if update.category_rule.strip():
            receptacle = _fuzzy_match_receptacle(
                update.category_rule_receptacle.strip(), state["receptacles"]
            )
            if not receptacle:
                receptacle = _fuzzy_match_receptacle(
                    (oracle_receptacle or "").strip(), state["receptacles"]
                )
            normalized = _normalize_confirmed_preference(
                preference=LearnedPreferenceModel(
                    hypothesis=update.category_rule.strip(),
                    covered_objects=update.category_rule_covered_objects,
                ),
                seen_objects=state["seen_objects"],
                # No fallback to proposer's covered_objects — the state-update LLM
                # determines which objects the rule covers based on the oracle's answer.
                # Proposer groupings are speculative and often wrong.
                receptacle=receptacle,
            )
            if normalized is not None:
                _upsert_confirmed_preference(state, normalized)
                if receptacle:
                    unresolved_set = set(state["unresolved_objects"])
                    for obj in normalized.get("covered_objects", []):
                        if obj in unresolved_set:
                            _upsert_confirmed_action(state, object_name=obj, receptacle=receptacle)
                else:
                    _auto_resolve_from_hypothesis(
                        state=state,
                        hypothesis_text=update.category_rule,
                        covered_objects=normalized.get("covered_objects", []),
                    )
        _apply_confirmed_actions(state=state, placements=update.confirmed_actions)
        _remove_negative_actions_for_confirmed(state=state, placements=update.confirmed_actions)
        _apply_negative_actions(state=state, placements=update.negative_actions)
        return recompute_online_placements(state)

    def update_state_from_preference_induction_answer(
        self,
        *,
        state: AgentState,
        hypothesis: str,
        covered_objects: Optional[List[str]],
        answer: str,
        question: Optional[str] = None,
    ) -> AgentState:
        interpretation = self.interpret_preference_induction_answer(
            state=state,
            hypothesis=hypothesis,
            covered_objects=covered_objects or [],
            answer=answer,
            question=question,
        )
        return self.apply_preference_induction_interpretation(
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


def _fuzzy_match_receptacle(text: str, receptacles: List[str]) -> str:
    """Return an exact receptacle name by progressive matching, or '' if no match.

    Priority:
    1. Exact match (case-insensitive)
    2. Extracted text is a substring of a receptacle name
    3. A receptacle name is a substring of the extracted text
    """
    if not text:
        return ""
    text_norm = text.strip().lower()
    for r in receptacles:
        if r.lower() == text_norm:
            return r
    for r in receptacles:
        if text_norm in r.lower():
            return r
    for r in receptacles:
        if r.lower() in text_norm:
            return r
    return ""


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
    receptacle: str = "",
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

    result = LearnedPreference(hypothesis=hypothesis, covered_objects=covered_objects)
    if receptacle:
        result["receptacle"] = receptacle
    return result


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


def _auto_resolve_from_hypothesis(
    *,
    state: AgentState,
    hypothesis_text: str,
    covered_objects: List[str],
) -> str:
    """If the hypothesis text mentions exactly one receptacle, resolve all covered objects to it.
    Returns the matched receptacle name, or "" if no unambiguous match."""
    matched = [r for r in state["receptacles"] if r in hypothesis_text]
    if len(matched) != 1:
        return ""
    receptacle = matched[0]
    unresolved_set = set(state["unresolved_objects"])
    for obj in covered_objects:
        if obj in unresolved_set:
            _upsert_confirmed_action(state, object_name=obj, receptacle=receptacle)
    return receptacle


def _apply_single_exception(
    *,
    state: AgentState,
    interpretation: PreferenceInductionInterpretation,
) -> None:
    """Apply a single exception object→receptacle placement from a preference_induction result."""
    obj = interpretation.exception_object_name.strip()
    rec = interpretation.exception_receptacle.strip()
    if not obj or not rec:
        return
    if obj not in state["seen_objects"] or rec not in state["receptacles"]:
        return
    _upsert_confirmed_action(state, object_name=obj, receptacle=rec)
    _remove_negative_action(state=state, object_name=obj, receptacle=rec)
