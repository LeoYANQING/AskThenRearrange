from __future__ import annotations

import os
from typing import Any, List, Literal, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.agent_schema import AgentState, QuestionPattern
except ModuleNotFoundError:
    from agent_schema import AgentState, QuestionPattern


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

PolicyMode = Literal[
    "direct_querying",
    "user_preference_first",
    "parallel_exploration",
    "hybrid_all",
]


class QuestionDecision(BaseModel):
    question_pattern: QuestionPattern = Field(description="The next question pattern to ask.")
    guidance: str = Field(
        description="A short natural-language hint for the downstream proposer about what kind of question to ask next."
    )


class QuestionPolicyController:
    """
    LLM-based high-level dialogue policy.

    Responsibilities:
    - read AgentState
    - choose the next question pattern
    - provide one short guidance string for the downstream proposer

    It does not generate the final question text itself. Pattern-specific proposers
    remain responsible for instantiating the actual question.
    """

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
            reasoning=False,
        )
        self.structured_model = self.model.with_structured_output(QuestionDecision)

    def plan_next_question(
        self,
        *,
        state: AgentState,
        mode: PolicyMode,
    ) -> Optional[QuestionDecision]:
        if _budget_used(state) >= state["budget_total"]:
            return None

        allowed_patterns = self._allowed_patterns(state=state, mode=mode)
        if not allowed_patterns:
            return None

        try:
            result = self.structured_model.invoke(
                [
                    {"role": "system", "content": self._system_prompt(mode=mode)},
                    {"role": "user", "content": self._user_prompt(state=state, allowed_patterns=allowed_patterns)},
                ]
            )
        except Exception:
            return self._fallback_decision(
                allowed_patterns=allowed_patterns,
                mode=mode,
            )

        if result.question_pattern not in allowed_patterns:
            return self._fallback_decision(
                allowed_patterns=allowed_patterns,
                mode=mode,
                guidance=result.guidance,
            )

        guidance = result.guidance.strip() or self._default_guidance(
            question_pattern=result.question_pattern,
            mode=mode,
        )
        return QuestionDecision(
            question_pattern=result.question_pattern,
            guidance=guidance,
        )

    def _fallback_decision(
        self,
        *,
        allowed_patterns: List[QuestionPattern],
        mode: PolicyMode,
        guidance: str = "",
    ) -> QuestionDecision:
        pattern = allowed_patterns[0]
        return QuestionDecision(
            question_pattern=pattern,
            guidance=guidance.strip() or self._default_guidance(
                question_pattern=pattern,
                mode=mode,
            ),
        )

    def _allowed_patterns(
        self,
        *,
        state: AgentState,
        mode: PolicyMode,
    ) -> List[QuestionPattern]:
        can_eliciting = bool(state["unresolved_objects"])
        can_action = bool(state["unresolved_objects"])
        can_summary = self._summary_is_available(state=state)

        if mode == "direct_querying":
            return ["action_oriented"] if can_action else []

        if mode == "user_preference_first":
            allowed: List[QuestionPattern] = []
            if can_eliciting:
                allowed.append("preference_eliciting")
            if can_action:
                allowed.append("action_oriented")
            return allowed

        if mode == "parallel_exploration":
            allowed: List[QuestionPattern] = []
            if can_action:
                allowed.append("action_oriented")
            if can_summary:
                allowed.append("preference_summary")
            return allowed

        if mode == "hybrid_all":
            allowed: List[QuestionPattern] = []
            if can_summary:
                allowed.append("preference_summary")
            if can_eliciting:
                allowed.append("preference_eliciting")
            if can_action:
                allowed.append("action_oriented")
            return allowed

        raise ValueError(f"Unsupported policy mode: {mode}")

    def _summary_is_available(self, *, state: AgentState) -> bool:
        summarized_objects = set()
        for item in _confirmed_preferences(state):
            summarized_objects.update(item.get("covered_objects", []))

        unsummarized_action_count = sum(
            1 for obj in _confirmed_action_objects(state) if obj not in summarized_objects
        )
        return len(state["confirmed_actions"]) >= 2 and unsummarized_action_count >= 2

    def _system_prompt(self, *, mode: PolicyMode) -> str:
        if mode == "direct_querying":
            strategy_block = """
Strategy: Direct Querying.
- Choose action_oriented to directly resolve one unresolved object's placement.
- Prefer the next question that is most likely to yield a concrete object-level placement.
- Do not choose preference_eliciting or preference_summary.
""".strip()
        elif mode == "user_preference_first":
            strategy_block = """
Strategy: User-Preference-First.
- Choose preference_eliciting when an open preference hypothesis could clarify placements for multiple unresolved objects.
- Choose action_oriented when unresolved objects are ambiguous under the current confirmed_preferences and an object-level question would better test a boundary or resolve a concrete uncertainty.
- If an open preference hypothesis is too narrow to affect only one unresolved object, prefer action_oriented instead.
- Do not choose preference_summary in this strategy.
""".strip()
        elif mode == "parallel_exploration":
            strategy_block = """
Strategy: parallel-exploration.
- Choose action_oriented when the next object-level answer would most usefully extend current evidence, test a partial pattern, or create support for a future summary.
- Prefer action questions that build on current confirmed_actions or confirmed_preferences rather than isolated one-off placements.
- Choose preference_summary when existing confirmed_actions already support a stable multi-object rule that is worth confirming.
- Do not choose preference_summary too early when the current evidence is still sparse, weak, or fragmented.
- Do not choose preference_eliciting in this strategy.
""".strip()
        else:
            strategy_block = """
Strategy: Hybrid-All.
- You may choose among preference_eliciting, action_oriented, and preference_summary.
- Choose preference_eliciting when a missing high-level preference could clarify placements for multiple unresolved objects.
- Choose action_oriented when uncertainty is concentrated on specific unresolved objects and would be better reduced by grounding a concrete placement or testing the boundary of an existing preference.
- Choose preference_summary when existing confirmed_actions already support a stable multi-object rule that is worth confirming.
- Do not choose preference_summary too early when the current action evidence is still sparse or weak.
- Do not default to any pattern solely because it is available.
- Choose the pattern that would reduce the most uncertainty in the current AgentState.
""".strip()

        return f"""
You are the high-level question policy controller for a household rearrangement agent.

Your job:
- choose exactly one next question pattern from the allowed patterns
- produce one short guidance string for the downstream proposer

The guidance should:
- be one sentence
- explain what the proposer should focus on next
- not be a full user-facing question
- help the proposer choose a good object, hypothesis, or summary

General rules:
- Use only the allowed question patterns.
- Respect the strategy instructions.
- Be conservative and state-driven.
- Base the decision only on the provided AgentState summary.
- If action_oriented is chosen, the guidance may suggest probing a boundary or collecting evidence for a future summary, but do not mention internal code concepts.
- If preference_eliciting is chosen, the guidance should emphasize unresolved high-level preferences.
- If preference_summary is chosen, the guidance should emphasize confirming or refining a summary that is already supported by evidence.

{strategy_block}

Return only structured output.
""".strip()

    def _user_prompt(
        self,
        *,
        state: AgentState,
        allowed_patterns: List[QuestionPattern],
    ) -> str:
        summarized_objects = set()
        for item in _confirmed_preferences(state):
            summarized_objects.update(item.get("covered_objects", []))

        unsummarized_action_count = sum(
            1 for obj in _confirmed_action_objects(state) if obj not in summarized_objects
        )
        recent_qa_history = state["qa_history"][-3:]
        history_patterns = [
            item.get("question_pattern")
            for item in recent_qa_history
            if item.get("question_pattern")
        ]
        last_pattern = history_patterns[-1] if history_patterns else None
        recent_pattern_streak = 0
        for pattern in reversed(history_patterns):
            if pattern != last_pattern:
                break
            recent_pattern_streak += 1

        return f"""
Allowed question patterns:
{allowed_patterns}

Derived state summary:
- budget_left: {max(0, state['budget_total'] - _budget_used(state))}
- num_unresolved: {len(state['unresolved_objects'])}
- num_confirmed_actions: {len(state['confirmed_actions'])}
- num_confirmed_preferences: {len(_confirmed_preferences(state))}
- num_unsummarized_actions: {unsummarized_action_count}
- last_pattern: {last_pattern}
- recent_pattern_streak: {recent_pattern_streak}

Current state:
- unresolved_objects: {state['unresolved_objects']}
- confirmed_actions: {state['confirmed_actions']}
- confirmed_preferences: {_confirmed_preferences(state)}
- negative_preferences: {state['negative_preferences']}
- recent_qa_history: {recent_qa_history}
""".strip()

    def _default_guidance(self, *, question_pattern: QuestionPattern, mode: PolicyMode) -> str:
        if question_pattern == "action_oriented":
            if mode == "user_preference_first":
                return "Ask an action question that checks the boundary of an already known preference or cleans up a remaining concrete placement."
            if mode == "parallel_exploration":
                return "Ask an action question that collects object-level evidence likely to support a future summary rule."
            return "Ask an action question that resolves one unresolved object's placement clearly."
        if question_pattern == "preference_eliciting":
            return "Ask about the most useful unresolved high-level preference that can affect multiple visible objects."
        return "Ask a summary question that confirms or refines a rule already supported by accumulated evidence."


__all__ = [
    "PolicyMode",
    "QuestionDecision",
    "QuestionPolicyController",
]


def _budget_used(state: AgentState) -> int:
    return len(state["qa_history"])


def _confirmed_preferences(state: AgentState) -> List[dict]:
    return state["confirmed_preferences"]


def _confirmed_action_objects(state: AgentState) -> List[str]:
    return [item["object_name"] for item in state["confirmed_actions"]]
