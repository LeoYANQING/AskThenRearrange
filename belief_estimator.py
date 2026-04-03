from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent_schema import AgentState


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class ObjectBelief(BaseModel):
    """Placement belief for one unresolved object."""

    object_name: str = Field(description="Exact object name from the unresolved list.")
    top_receptacles: List[str] = Field(
        description="Top 3 most likely receptacles from the available list, ordered by probability.",
        min_length=1,
        max_length=5,
    )
    probabilities: List[float] = Field(
        description=(
            "Probability for each entry in top_receptacles. "
            "Values between 0 and 1, summing to at most 1.0. "
            "The remainder is spread across all other receptacles."
        ),
        min_length=1,
        max_length=5,
    )


class BeliefEstimate(BaseModel):
    """Belief estimates for all target objects in one call."""

    beliefs: List[ObjectBelief] = Field(
        description="One entry per target object (both seen-unresolved and unseen)."
    )


# ---------------------------------------------------------------------------
# Entropy utilities
# ---------------------------------------------------------------------------

def shannon_entropy(probs: List[float], num_total_bins: int) -> float:
    """Compute Shannon entropy (bits) from top-k probs + uniform remainder."""
    clamped = [max(0.0, min(1.0, p)) for p in probs]
    total = sum(clamped)
    if total > 1.0:
        clamped = [p / total for p in clamped]
        total = 1.0

    remainder = max(0.0, 1.0 - total)
    num_other = max(1, num_total_bins - len(clamped))
    other_p = remainder / num_other

    dist = clamped + [other_p] * num_other
    h = 0.0
    for p in dist:
        if p > 0:
            h -= p * math.log2(p)
    return h


def max_entropy(num_bins: int) -> float:
    """Maximum possible entropy for a uniform distribution over num_bins."""
    return math.log2(num_bins) if num_bins > 1 else 0.0


# ---------------------------------------------------------------------------
# BeliefEstimator
# ---------------------------------------------------------------------------

class BeliefEstimator:
    """
    Uses an LLM to estimate placement beliefs for unresolved objects,
    then computes Shannon entropy as an uncertainty signal.
    """

    def __init__(
        self,
        model: str = QUESTION_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.llm: Any = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
        )
        self.structured_llm = self.llm.with_structured_output(BeliefEstimate)

    def estimate(self, state: AgentState) -> Dict[str, float]:
        """
        Return {object_name: entropy} for every unresolved object.

        High entropy  → uncertain placement  → valuable to ask about.
        Low entropy   → confident placement  → less urgent.
        """
        if not state["unresolved_objects"]:
            return {}

        num_r = len(state["receptacles"])
        h_max = max_entropy(num_r)

        try:
            result = self.structured_llm.invoke([
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(state)},
            ])
        except Exception:
            return {obj: h_max for obj in state["unresolved_objects"]}

        return self._to_entropies(result, state)

    def estimate_detailed(
        self,
        state: AgentState,
        include_unseen: bool = False,
    ) -> Optional[BeliefEstimate]:
        """Return the full structured belief (for debugging / visualization).

        When include_unseen=True, also estimates beliefs for unseen objects
        so that generalization value can be computed.
        """
        target_objects = list(state["unresolved_objects"])
        if include_unseen:
            target_objects = target_objects + list(state["unseen_objects"])
        if not target_objects:
            return BeliefEstimate(beliefs=[])
        try:
            return self.structured_llm.invoke([
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(
                    state, target_objects=target_objects,
                )},
            ])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return """You are a household placement analyst.

Given the current knowledge about a user's object placements and preferences,
estimate the most likely receptacle for each unresolved object.

For EACH unresolved object, output:
- object_name: the exact name from the unresolved list
- top_receptacles: the 3 most likely receptacles (from the available receptacles list), ordered by likelihood
- probabilities: a probability for each, summing to at most 1.0

Calibration rules — you MUST vary probabilities based on your actual confidence:
- OBVIOUS placement (e.g. "hardcover novel" → bookshelf): top probability 0.70–0.90
- LIKELY but not certain (e.g. "remote" → TV stand or coffee table): top probability 0.45–0.65
- GENUINELY UNCERTAIN (e.g. "seasonal garland" — could go anywhere): top probability 0.25–0.35, spread evenly

Additional guidelines:
- If a confirmed_preference clearly applies, assign 0.70+ to the matching receptacle.
- If confirmed_actions show a pattern for similar objects, use that pattern (0.60+).
- Do NOT use the same probability pattern for every object. Each object has different certainty.
- Use only receptacles from the available receptacles list.

Return structured output only."""

    def _user_prompt(
        self,
        state: AgentState,
        target_objects: Optional[List[str]] = None,
    ) -> str:
        objects = target_objects or state["unresolved_objects"]
        return f"""Available receptacles: {state['receptacles']}

Target objects (estimate placement for each):
{objects}

Current evidence:
- confirmed_actions: {state['confirmed_actions']}
- confirmed_preferences: {state['confirmed_preferences']}
- negative_preferences: {state['negative_preferences']}
- recent_qa_history: {state['qa_history'][-5:] if state['qa_history'] else 'none'}"""

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    def _to_entropies(
        self, estimate: BeliefEstimate, state: AgentState
    ) -> Dict[str, float]:
        num_r = len(state["receptacles"])
        h_max = max_entropy(num_r)
        entropies: Dict[str, float] = {}

        covered = set()
        for belief in estimate.beliefs:
            covered.add(belief.object_name)
            entropies[belief.object_name] = shannon_entropy(
                belief.probabilities, num_r
            )

        # Objects the LLM missed → assign max entropy
        for obj in state["unresolved_objects"]:
            if obj not in covered:
                entropies[obj] = h_max

        return entropies


__all__ = [
    "BeliefEstimate",
    "BeliefEstimator",
    "ObjectBelief",
    "max_entropy",
    "shannon_entropy",
]
