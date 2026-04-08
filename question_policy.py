from __future__ import annotations

import os
from typing import Any, List, Literal, Optional

from llm_factory import create_chat_model, DEFAULT_MODEL, DEFAULT_BASE_URL
from pydantic import BaseModel, Field

from agent_schema import AgentState, QuestionPattern
from belief_estimator import BeliefEstimator, max_entropy, shannon_entropy


# Model config now in llm_factory.py
# Base URL config now in llm_factory.py

PolicyMode = Literal[
    "direct_querying",
    "user_preference_first",
    "parallel_exploration",
    "hybrid_all",
]

SelectionMethod = Literal["rule", "entropy", "llm"]


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
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        selection_method: SelectionMethod = "rule",
    ) -> None:
        self.selection_method = selection_method
        self.model: Any = create_chat_model(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
        )
        self.structured_model = self.model.with_structured_output(QuestionDecision)
        # Lazy-init: only create BeliefEstimator when entropy method is selected
        self._belief_estimator: Optional[BeliefEstimator] = None
        if selection_method == "entropy":
            self._belief_estimator = BeliefEstimator(
                model=model, base_url=base_url, temperature=temperature,
            )

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

        # ── Entropy-driven selection (applies to ALL modes) ──
        if self.selection_method == "entropy":
            return self._entropy_select(
                state=state, allowed_patterns=allowed_patterns, mode=mode,
            )

        # ── Rule-based selection ──
        if self.selection_method == "rule":
            if mode == "direct_querying":
                return self._rule_direct_querying()
            if mode == "user_preference_first":
                return self._rule_user_preference_first(state=state, allowed_patterns=allowed_patterns)
            if mode == "parallel_exploration":
                return self._rule_parallel_exploration(state=state, allowed_patterns=allowed_patterns)
            if mode == "hybrid_all":
                return self._rule_hybrid_all(state=state, allowed_patterns=allowed_patterns)

        # ── LLM-based selection (original hybrid_all controller) ──
        return self._llm_select(
            state=state, allowed_patterns=allowed_patterns, mode=mode,
        )

    # ------------------------------------------------------------------
    # Rule-based policy decisions (no LLM call)
    # ------------------------------------------------------------------

    def _rule_direct_querying(self) -> QuestionDecision:
        return QuestionDecision(
            question_pattern="action_oriented",
            guidance=self._default_guidance(
                question_pattern="action_oriented", mode="direct_querying"
            ),
        )

    def _rule_user_preference_first(
        self,
        *,
        state: AgentState,
        allowed_patterns: List[QuestionPattern],
    ) -> QuestionDecision:
        covered_by_prefs: set = set()
        for p in _confirmed_preferences(state):
            covered_by_prefs.update(p.get("covered_objects", []))
        uncovered = [
            obj for obj in state["unresolved_objects"] if obj not in covered_by_prefs
        ]

        covered_recs = {p.get("receptacle") for p in _confirmed_preferences(state) if p.get("receptacle")}
        covered_recs |= {a.get("receptacle") for a in state["confirmed_actions"] if a.get("receptacle")}
        uncovered_recs = [r for r in state["receptacles"] if r not in covered_recs]

        # Count how many PE and AO questions have been asked
        pe_count = sum(1 for qa in state["qa_history"] if qa.get("question_pattern") == "preference_eliciting")
        ao_count = sum(1 for qa in state["qa_history"] if qa.get("question_pattern") == "action_oriented")
        total = len(state["qa_history"])
        num_cps = len(state["confirmed_preferences"])

        # Strategy: PE first to cover all receptacles, then AO to probe boundaries
        # Only switch to AO boundary probe when all receptacles have at least a CP or CA
        if not uncovered_recs and num_cps >= 2 and "action_oriented" in allowed_patterns and len(uncovered) >= 1:
            cp_summaries = []
            for cp in state["confirmed_preferences"]:
                h = cp.get("hypothesis", "")
                r = cp.get("receptacle", "")
                if h and r:
                    cp_summaries.append(f'"{h}" → {r}')
            cp_text = "; ".join(cp_summaries)
            ca_recs = sorted({a.get("receptacle") for a in state["confirmed_actions"] if a.get("receptacle")})
            return QuestionDecision(
                question_pattern="action_oriented",
                guidance=(
                    f"All receptacles are covered. Learned rules: {cp_text}. "
                    f"Now ask about an unresolved object that is AMBIGUOUS — "
                    f"it could fit more than one rule or sits at the boundary between categories. "
                    f"Receptacles with confirmed actions: {ca_recs}. "
                    f"Prefer an object likely to go to a receptacle NOT in {ca_recs}."
                ),
            )

        # Otherwise: PE if there are uncovered objects/receptacles
        if "preference_eliciting" in allowed_patterns and (len(uncovered) >= 2 or uncovered_recs):
            pattern: QuestionPattern = "preference_eliciting"
        else:
            pattern = "action_oriented"
        return QuestionDecision(
            question_pattern=pattern,
            guidance=self._default_guidance(
                question_pattern=pattern, mode="user_preference_first"
            ),
        )

    def _rule_parallel_exploration(
        self,
        *,
        state: AgentState,
        allowed_patterns: List[QuestionPattern],
    ) -> QuestionDecision:
        # PAR = AO + PI only (no PE). AO collects evidence, PI induces rules.
        CONSOLIDATE_AFTER = 2

        # Receptacle coverage analysis
        cp_recs = {cp.get("receptacle") for cp in state["confirmed_preferences"] if cp.get("receptacle")}
        ca_recs_count: dict = {}
        for ca in state["confirmed_actions"]:
            r = ca.get("receptacle", "")
            if r:
                ca_recs_count[r] = ca_recs_count.get(r, 0) + 1
        all_covered = cp_recs | set(ca_recs_count.keys())
        uncovered = sorted(r for r in state["receptacles"] if r not in all_covered)
        # Receptacles with only 1 CA and no CP — weak coverage, need more AO evidence
        weak_recs = sorted(r for r in state["receptacles"] if r not in cp_recs and ca_recs_count.get(r, 0) <= 1)

        # Count consecutive AO turns since last PI
        since_last_pi = 0
        for item in reversed(state["qa_history"]):
            if item.get("question_pattern") == "preference_induction":
                break
            since_last_pi += 1

        # PI: after CONSOLIDATE_AFTER consecutive AOs, try to induce a rule
        # BUT only for receptacles that actually have ≥2 CAs (evidence-backed induction)
        if since_last_pi >= CONSOLIDATE_AFTER and "preference_induction" in allowed_patterns:
            # Find receptacles with enough AO evidence but no CP yet
            recs_with_evidence = sorted(
                r for r, count in ca_recs_count.items()
                if count >= 2 and r not in cp_recs
            )
            if recs_with_evidence:
                # Show the actual CA evidence to ground the PI
                evidence_summary = []
                for r in recs_with_evidence[:2]:
                    objs = [ca["object_name"] for ca in state["confirmed_actions"] if ca.get("receptacle") == r]
                    evidence_summary.append(f"{r}: {objs}")
                pi_guidance = (
                    f"Induce a BROAD placement rule from these confirmed actions: {evidence_summary}. "
                    f"The rule should describe the GENERAL CATEGORY of items at that receptacle, "
                    f"not just repeat the specific objects. Make it broad enough to cover similar unseen items."
                )
                return QuestionDecision(
                    question_pattern="preference_induction",
                    guidance=pi_guidance,
                )
            # No receptacle has enough evidence for PI — do another AO instead
            # (fall through to AO below)

        # AO strategy: explore uncovered/weak receptacles, avoid receptacles with CP
        already_covered = sorted(cp_recs)
        if uncovered:
            ao_guidance = (
                f"Receptacles with NO evidence yet: {uncovered}. "
                f"Receptacles to AVOID (already have rules): {already_covered}. "
                f"Pick an unresolved object most likely to belong to one of the uncovered receptacles. "
                f"Do NOT pick an object that obviously belongs to {already_covered}."
            )
        elif weak_recs:
            ao_guidance = (
                f"Receptacles needing more evidence (≤1 CA, no rule yet): {weak_recs}. "
                f"Receptacles to AVOID (already have rules): {already_covered}. "
                f"Pick an unresolved object likely to belong to one of the weak receptacles. "
                f"A second data point at {weak_recs} enables rule induction."
            )
        else:
            # All receptacles have some coverage — find ones with CP but few CAs
            thin_recs = sorted(r for r in state["receptacles"] if ca_recs_count.get(r, 0) <= 1)
            if thin_recs:
                ao_guidance = (
                    f"All receptacles have rules, but these have thin evidence (≤1 CA): {thin_recs}. "
                    f"Pick an object likely to belong to one of these to strengthen the evidence."
                )
            else:
                ao_guidance = "All receptacles well covered. Pick any remaining unresolved object."
        return QuestionDecision(
            question_pattern="action_oriented",
            guidance=ao_guidance,
        )

    def _rule_hybrid_all(
        self,
        *,
        state: AgentState,
        allowed_patterns: List[QuestionPattern],
    ) -> QuestionDecision:
        # Merge user_preference_first + parallel_exploration rules:
        # 1st priority: preference_eliciting if uncovered ≥ 2
        covered_by_prefs: set = set()
        for p in _confirmed_preferences(state):
            covered_by_prefs.update(p.get("covered_objects", []))
        uncovered = [
            obj for obj in state["unresolved_objects"] if obj not in covered_by_prefs
        ]
        if "preference_eliciting" in allowed_patterns and len(uncovered) >= 2:
            return QuestionDecision(
                question_pattern="preference_eliciting",
                guidance=self._default_guidance(
                    question_pattern="preference_eliciting", mode="hybrid_all"
                ),
            )
        # 2nd priority: preference_induction if unsummarized ≥ 3
        summarized: set = set()
        for p in _confirmed_preferences(state):
            summarized.update(p.get("covered_objects", []))
        unsummarized_count = sum(
            1 for obj in _confirmed_action_objects(state) if obj not in summarized
        )
        if "preference_induction" in allowed_patterns and unsummarized_count >= 3:
            return QuestionDecision(
                question_pattern="preference_induction",
                guidance=self._default_guidance(
                    question_pattern="preference_induction", mode="hybrid_all"
                ),
            )
        # Fallback: action_oriented
        return QuestionDecision(
            question_pattern="action_oriented",
            guidance=self._default_guidance(
                question_pattern="action_oriented", mode="hybrid_all"
            ),
        )

    # ------------------------------------------------------------------
    # Entropy-driven selection via Expected Entropy Reduction (EER)
    # ------------------------------------------------------------------

    def _entropy_select(
        self,
        *,
        state: AgentState,
        allowed_patterns: List[QuestionPattern],
        mode: PolicyMode,
    ) -> QuestionDecision:
        assert self._belief_estimator is not None

        # Estimate beliefs for both seen (unresolved) and unseen objects
        needs_unseen = (
            "preference_induction" in allowed_patterns
            or "preference_eliciting" in allowed_patterns
        )
        belief_result = self._belief_estimator.estimate_detailed(
            state, include_unseen=needs_unseen,
        )
        if belief_result is None or not belief_result.beliefs:
            return self._fallback_decision(allowed_patterns=allowed_patterns, mode=mode)

        num_r = len(state["receptacles"])
        h_max = max_entropy(num_r)

        # ── Build per-object entropy and top-1 receptacle maps ──
        seen_set = set(state["unresolved_objects"])
        unseen_set = set(state["unseen_objects"])

        seen_entropy: dict[str, float] = {}
        seen_top1: dict[str, str] = {}
        unseen_top1: dict[str, str] = {}
        unseen_entropy: dict[str, float] = {}

        for b in belief_result.beliefs:
            h = shannon_entropy(b.probabilities, num_r)
            top1 = b.top_receptacles[0] if b.top_receptacles else ""
            if b.object_name in seen_set:
                seen_entropy[b.object_name] = h
                seen_top1[b.object_name] = top1
            elif b.object_name in unseen_set:
                unseen_entropy[b.object_name] = h
                unseen_top1[b.object_name] = top1

        for obj in state["unresolved_objects"]:
            if obj not in seen_entropy:
                seen_entropy[obj] = h_max

        # ── EER: action_oriented ──
        # Resolves one seen object; no generalization to unseen.
        highest_obj = max(seen_entropy, key=seen_entropy.get)  # type: ignore[arg-type]
        eer_action = seen_entropy[highest_obj]

        # ── EER: preference_eliciting ──
        # Group seen objects by top-1 receptacle.
        # Generalization bonus: count unseen objects with same top-1.
        seen_groups: dict[str, list[tuple[str, float]]] = {}
        for obj, h in seen_entropy.items():
            top1 = seen_top1.get(obj, "")
            if top1:
                seen_groups.setdefault(top1, []).append((obj, h))

        unseen_by_receptacle: dict[str, int] = {}
        for obj, top1 in unseen_top1.items():
            if top1:
                unseen_by_receptacle[top1] = unseen_by_receptacle.get(top1, 0) + 1

        # Generalization weights:
        # - Eliciting asks about hypothetical preferences (speculative) → α = 0.5
        # - Induction confirms observed action patterns (evidence-backed) → α = 1.0
        ALPHA_ELICITING = 0.5
        ALPHA_INDUCTION = 1.0

        eer_eliciting = 0.0
        best_group_receptacle = ""
        best_group_objects: list[str] = []
        best_group_unseen_count = 0
        for receptacle, members in seen_groups.items():
            if len(members) >= 2:
                seen_h = sum(h for _, h in members)
                n_unseen = unseen_by_receptacle.get(receptacle, 0)
                total_eer = seen_h + ALPHA_ELICITING * n_unseen * h_max
                if total_eer > eer_eliciting:
                    eer_eliciting = total_eer
                    best_group_receptacle = receptacle
                    best_group_objects = [obj for obj, _ in members]
                    best_group_unseen_count = n_unseen

        # ── EER: preference_induction ──
        # Evidence-backed: confirmed actions prove a pattern exists.
        # Higher generalization weight because the rule is grounded in observation.
        eer_induction = 0.0
        induction_receptacle = ""
        induction_unseen_count = 0
        if self._induction_is_available(state=state):
            summarized: set[str] = set()
            for p in _confirmed_preferences(state):
                summarized.update(p.get("covered_objects", []))
            unsummarized = [
                a for a in state["confirmed_actions"]
                if a["object_name"] not in summarized
            ]
            if unsummarized:
                from collections import Counter
                receptacle_counts = Counter(a["receptacle"] for a in unsummarized)
                dominant_receptacle = receptacle_counts.most_common(1)[0][0]
                induction_receptacle = dominant_receptacle
                n_actions = receptacle_counts[dominant_receptacle]

                # Seen objects matching
                seen_h = sum(
                    h for obj, h in seen_entropy.items()
                    if seen_top1.get(obj) == dominant_receptacle
                )
                # Unseen generalization bonus (evidence-backed → full weight)
                n_unseen = unseen_by_receptacle.get(dominant_receptacle, 0)
                eer_induction = seen_h + ALPHA_INDUCTION * n_unseen * h_max
                induction_unseen_count = n_unseen

        # ── Select pattern with highest EER ──
        candidates: list[tuple[QuestionPattern, float, str]] = []

        if "action_oriented" in allowed_patterns:
            candidates.append((
                "action_oriented",
                eer_action,
                f"Focus on '{highest_obj}' — highest placement uncertainty ({eer_action:.2f} bits).",
            ))

        if "preference_eliciting" in allowed_patterns and eer_eliciting > 0:
            top_names = ", ".join(best_group_objects[:3])
            unseen_note = f", +{best_group_unseen_count} unseen" if best_group_unseen_count else ""
            candidates.append((
                "preference_eliciting",
                eer_eliciting,
                f"Objects like {top_names} share uncertainty toward '{best_group_receptacle}' "
                f"(EER={eer_eliciting:.2f} bits, {len(best_group_objects)} seen{unseen_note}). "
                f"Ask a preference covering this group.",
            ))

        if "preference_induction" in allowed_patterns and eer_induction > 0:
            unseen_note = f", +{induction_unseen_count} unseen" if induction_unseen_count else ""
            candidates.append((
                "preference_induction",
                eer_induction,
                f"Actions suggest a pattern around '{induction_receptacle}' "
                f"(EER={eer_induction:.2f} bits{unseen_note}). Confirm or refine this rule.",
            ))

        if not candidates:
            return self._fallback_decision(allowed_patterns=allowed_patterns, mode=mode)

        best_pattern, _, best_guidance = max(candidates, key=lambda x: x[1])
        return QuestionDecision(question_pattern=best_pattern, guidance=best_guidance)

    # ------------------------------------------------------------------
    # LLM-based selection (original controller)
    # ------------------------------------------------------------------

    def _llm_select(
        self,
        *,
        state: AgentState,
        allowed_patterns: List[QuestionPattern],
        mode: PolicyMode,
    ) -> QuestionDecision:
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
        has_unresolved = bool(state["unresolved_objects"])
        # Allow PE even without unresolved objects (receptacle-centric questions don't need them)
        covered_recs = {p.get("receptacle") for p in _confirmed_preferences(state) if p.get("receptacle")}
        covered_recs |= {a.get("receptacle") for a in state["confirmed_actions"] if a.get("receptacle")}
        has_uncovered_recs = any(r not in covered_recs for r in state["receptacles"])
        can_eliciting = has_unresolved or has_uncovered_recs
        # Allow AO even when unresolved is empty — can still ask boundary probes about seen objects
        can_action = True
        can_induction = self._induction_is_available(state=state)

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
            # PAR = AO + PI only (no PE)
            allowed: List[QuestionPattern] = []
            if can_action:
                allowed.append("action_oriented")
            if can_induction:
                allowed.append("preference_induction")
            return allowed

        if mode == "hybrid_all":
            allowed: List[QuestionPattern] = []
            if can_induction:
                allowed.append("preference_induction")
            if can_eliciting:
                allowed.append("preference_eliciting")
            if can_action:
                allowed.append("action_oriented")
            return allowed

        raise ValueError(f"Unsupported policy mode: {mode}")

    def _induction_is_available(self, *, state: AgentState) -> bool:
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
- Do not choose preference_eliciting or preference_induction.
""".strip()
        elif mode == "user_preference_first":
            strategy_block = """
Strategy: User-Preference-First.
- Choose preference_eliciting when an open preference hypothesis could clarify placements for multiple unresolved objects.
- Choose action_oriented when unresolved objects are ambiguous under the current confirmed_preferences and an object-level question would better test a boundary or resolve a concrete uncertainty.
- If an open preference hypothesis is too narrow to affect only one unresolved object, prefer action_oriented instead.
- Do not choose preference_induction in this strategy.
""".strip()
        elif mode == "parallel_exploration":
            strategy_block = """
Strategy: parallel-exploration.
- Choose action_oriented when the next object-level answer would most usefully extend current evidence, test a partial pattern, or create support for a future summary.
- Prefer action questions that build on current confirmed_actions or confirmed_preferences rather than isolated one-off placements.
- Choose preference_induction when existing confirmed_actions already support a stable multi-object rule that is worth confirming.
- Do not choose preference_induction too early when the current evidence is still sparse, weak, or fragmented.
- Do not choose preference_eliciting in this strategy.
""".strip()
        else:
            strategy_block = """
Strategy: Hybrid-All.
- You may choose among preference_eliciting, action_oriented, and preference_induction.
- Choose preference_eliciting when a missing high-level preference could clarify placements for multiple unresolved objects.
- Choose action_oriented when uncertainty is concentrated on specific unresolved objects and would be better reduced by grounding a concrete placement or testing the boundary of an existing preference.
- Choose preference_induction when existing confirmed_actions already support a stable multi-object rule that is worth confirming.
- Do not choose preference_induction too early when the current action evidence is still sparse or weak.
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
- If preference_induction is chosen, the guidance should emphasize confirming or refining a summary that is already supported by evidence.

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
                return "Ask an action question to collect one more concrete placement — building toward the evidence needed for the next induction step."
            return "Ask an action question that resolves one unresolved object's placement clearly."
        if question_pattern == "preference_eliciting":
            return "Ask about the most useful unresolved high-level preference that can affect multiple visible objects."
        return "Ask a summary question that confirms or refines a rule already supported by accumulated evidence."


__all__ = [
    "PolicyMode",
    "QuestionDecision",
    "QuestionPolicyController",
    "SelectionMethod",
]


def _budget_used(state: AgentState) -> int:
    return len(state["qa_history"])


def _confirmed_preferences(state: AgentState) -> List[dict]:
    return state["confirmed_preferences"]


def _confirmed_action_objects(state: AgentState) -> List[str]:
    return [item["object_name"] for item in state["confirmed_actions"]]
