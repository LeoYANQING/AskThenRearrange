from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any, List, Optional

from langchain_ollama import ChatOllama

try:
    from v2.agent_schema import (
        ActionIntent,
        AgentState,
        PreferenceRecord,
        PreferenceElicitingIntent,
        PreferenceElicitingIntentBatch,
        PreferenceSummaryIntent,
        PreferenceSummaryIntentBatch,
    )
    from v2.data import get_episode
    from v2.state_init import build_initial_state
    from v2.state_update import StateUpdate
except ModuleNotFoundError:
    from agent_schema import (
        ActionIntent,
        AgentState,
        PreferenceRecord,
        PreferenceElicitingIntent,
        PreferenceElicitingIntentBatch,
        PreferenceSummaryIntent,
        PreferenceSummaryIntentBatch,
    )
    from data import get_episode
    from state_init import build_initial_state
    from state_update import StateUpdate


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "scenarios_aug_tiny.json"


# =========================================================
# Shared helpers
# =========================================================

def _clip_priority(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _existing_preference_texts(state: AgentState) -> set[str]:
    existing = set()
    for item in state["preference_candidates"]:
        text = item.get("hypothesis", "").strip().lower()
        if text:
            existing.add(text)
    for item in state["confirmed_preferences"]:
        text = item.get("hypothesis", "").strip().lower()
        if text:
            existing.add(text)
    for item in state["rejected_hypotheses"]:
        text = item.strip().lower()
        if text:
            existing.add(text)
    return existing


def _upsert_preference_candidate(
    *,
    state: AgentState,
    hypothesis: str,
    covered_objects: List[str],
) -> None:
    hypothesis_key = hypothesis.strip().lower()
    if not hypothesis_key:
        return

    candidate = PreferenceRecord(
        hypothesis=hypothesis,
        source="induced",
        covered_objects=covered_objects,
        target_receptacle=None,
        exceptions=[],
    )
    for idx, existing in enumerate(state["preference_candidates"]):
        if existing.get("hypothesis", "").strip().lower() == hypothesis_key:
            state["preference_candidates"][idx] = candidate
            return
    state["preference_candidates"].append(candidate)


# =========================================================
# Preference-eliciting proposer
# =========================================================

class PreferenceElicitingProposer:
    """
    Scene-derived proposer for Preference-eliciting questions.

    Input:
    - room
    - receptacles
    - seen_objects
    Output:
    - a small list of PreferenceElicitingIntent
    - each intent already includes a natural question
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
        self.structured_model = self.model.with_structured_output(
            PreferenceElicitingIntentBatch
        )

    def propose(
        self,
        *,
        state: AgentState,
        max_intents: int = 4,
        guidance: str = "",
    ) -> List[PreferenceElicitingIntent]:
        system_prompt = f"""
You are a proposer for preference-eliciting questions in a household rearrangement task.

Your job:
Propose a small number of high-value preference questions that are worth spending budget on.

Choose questions whose answers are most likely to:
- clarify placements for multiple unresolved objects
- reveal a stable placement rule or strong storage preference
- reduce uncertainty not already explained by confirmed_preferences

Avoid questions that:
- only affect one unresolved object unless no broader question remains
- restate an already confirmed preference
- ask directly for a specific object placement

Rules:
- This pattern is always "preference_eliciting".
- Do not output action-oriented questions.
- Do not output preference-summary questions.
- Return at most {max_intents} intents.
- Each intent must already include a natural user-facing question.
- Use only exact seen object names in covered_objects.
- Ask about high-level preferences, not a direct final placement.
""".strip()

        user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Open preference hypotheses:
{state["open_preference_hypotheses"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Guidance:
{guidance}

Choose the most useful unresolved preference question, not just a plausible one.
Use open_preference_hypotheses as candidate directions, but prioritize the one with the highest expected impact on unresolved objects.

Each intent must include:
- question_pattern = "preference_eliciting"
- hypothesis = a concise high-level preference hypothesis
- covered_objects = exact seen objects related to that hypothesis
- priority = 0.0 to 1.0
- question = one concise natural question directly asking that preference
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_preference_eliciting_intents(
            intents=result.intents,
            state=state,
            max_intents=max_intents,
        )


def _normalize_preference_eliciting_intents(
    *,
    intents: List[PreferenceElicitingIntent],
    state: AgentState,
    max_intents: int,
) -> List[PreferenceElicitingIntent]:
    allowed_objects = set(state["seen_objects"])
    open_hypotheses = {item.strip().lower() for item in state["open_preference_hypotheses"] if item.strip()}
    normalized: List[PreferenceElicitingIntent] = []
    seen_signatures = set()

    for item in intents:
        hypothesis = item.hypothesis.strip()
        question = item.question.strip()
        if not hypothesis or not question:
        #     continue
        # if open_hypotheses and hypothesis.lower() not in open_hypotheses:
            continue

        covered_objects = [
            obj for obj in _dedupe_keep_order(list(item.covered_objects))
            if obj in allowed_objects
        ]
        if not covered_objects:
            continue

        signature = (
            hypothesis.lower(),
            tuple(sorted(covered_objects)),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        normalized.append(
            PreferenceElicitingIntent(
                question_pattern="preference_eliciting",
                hypothesis=hypothesis,
                covered_objects=covered_objects,
                priority=_clip_priority(item.priority),
                question=question,
            )
        )

    normalized.sort(key=lambda x: x.priority, reverse=True)
    return normalized[:max_intents]


# =========================================================
# Action-oriented proposer
# =========================================================

class ActionProposer:
    """
    Proposer for Action-oriented questions.

    This proposer covers two internal modes:
    - direct_grounding
    - boundary_probe

    Both remain within the same empirical pattern:
    - question_pattern = "action_oriented"
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
        self.structured_model = self.model.with_structured_output(ActionIntent)

    def propose(
        self,
        *,
        state: AgentState,
        guidance: str = "",
    ) -> Optional[ActionIntent]:
        if not state["unresolved_objects"]:
            return None

        system_prompt = """
You are a proposer for Action-oriented questions in a household rearrangement task.

Your job:
Choose ONE unresolved seen object that is currently the best next target for an Action-oriented question.

Important:
- This empirical pattern is always "action_oriented".
- There are two internal action modes:
  1) direct_grounding:
     ask where a specific unresolved object should go
  2) boundary_probe:
     ask about a specific unresolved object in order to test whether an already known preference rule extends to it
- boundary_probe is NOT a fourth question pattern. It is only a subtype of Action-oriented questioning.

Rules:
- Output exactly one ActionIntent.
- The chosen object must be an exact unresolved seen object.
- Do not choose an already confirmed object.
- The output must already include a natural user-facing question.
- The question must explicitly ask about the placement or boundary status of the chosen object.
- The question must mention the chosen object by name.
- Do not ask about purchasing, planning, unrelated tasks, or any object other than the chosen object.
- Be conservative and stable.
- Prefer boundary_probe only when there is a plausible confirmed preference whose boundary can be tested with the object.
- Use the guidance as a soft instruction for whether this turn should probe a boundary, clean up a concrete placement, or collect evidence that could support a future summary.
""".strip()

        recent_qa_history = state["qa_history"][-3:]

        user_prompt = f"""
Unresolved seen objects:
{state["unresolved_objects"]}

Confirmed actions:
{state["confirmed_actions"]}

Preference candidates:
{state["preference_candidates"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Guidance:
{guidance}

Recent QA history:
{recent_qa_history}

Return exactly one ActionIntent:
- question_pattern = "action_oriented"
- action_mode = "direct_grounding" or "boundary_probe"
- object_name = one exact unresolved seen object
- priority = 0.0 to 1.0
- question = one concise natural Action-oriented question about that exact object's placement or boundary case
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_action_intent(
            intent=result,
            state=state,
        )


def _normalize_action_intent(
    *,
    intent: ActionIntent,
    state: AgentState,
) -> Optional[ActionIntent]:
    object_name = intent.object_name.strip()
    question = intent.question.strip()

    if not object_name or not question:
        return None
    if object_name not in state["unresolved_objects"]:
        return None
    if object_name in state["confirmed_actions"]:
        return None

    action_mode = intent.action_mode
    if action_mode not in ("direct_grounding", "boundary_probe"):
        action_mode = "direct_grounding"

    return ActionIntent(
        question_pattern="action_oriented",
        action_mode=action_mode,
        object_name=object_name,
        priority=_clip_priority(intent.priority),
        question=question,
    )


# =========================================================
# Preference-summary proposer
# =========================================================

class PreferenceSummaryProposer:
    """
    Proposer for Preference-summary questions.

    These questions summarize a candidate rule inferred from existing evidence
    and ask the user to confirm or refine it.
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
        self.structured_model = self.model.with_structured_output(
            PreferenceSummaryIntentBatch
        )

    def propose(
        self,
        *,
        state: AgentState,
        max_intents: int = 3,
        guidance: str = "",
    ) -> List[PreferenceSummaryIntent]:

        system_prompt = f"""
You are a proposer for preference-summary questions in a household rearrangement task.

Your job:
Propose a small number of high-value summary questions that are worth spending budget on.

Choose summaries whose confirmation is most likely to:
- explain multiple confirmed_actions or unresolved objects
- compress existing object-level evidence into a useful placement rule
- reduce uncertainty not already resolved by confirmed_preferences

Avoid summaries that:
- restate an already confirmed preference
- are too weak or too narrow to affect future placement decisions
- only describe one object unless no broader summary remains

Rules:
- This pattern is always "preference_summary".
- Do not output action-oriented questions.
- Do not output preference-eliciting questions.
- Return at most {max_intents} intents.
- Each intent must already include a natural user-facing summary question.
- Use only exact seen object names in covered_objects.
- Use the guidance as a soft instruction about what kind of summary is most useful to confirm next.
""".strip()

        user_prompt = f"""
Confirmed actions:
{state["confirmed_actions"]}

Seen objects:
{state["seen_objects"]}

Unresolved objects:
{state["unresolved_objects"]}

Existing preference candidates:
{state["preference_candidates"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Rejected hypotheses:
{state["rejected_hypotheses"]}

Guidance:
{guidance}

Choose the most useful summary question, not just a plausible one.
Use confirmed_actions as the main evidence source, and prefer a summary whose confirmation would improve future placement decisions.

Each intent must include:
- question_pattern = "preference_summary"
- hypothesis = a concise summary rule to confirm
- covered_objects = exact seen objects plausibly covered by the summary
- priority = 0.0 to 1.0
- question = one concise natural summary question
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_preference_summary_intents(
            intents=result.intents,
            state=state,
            max_intents=max_intents,
        )


def _normalize_preference_summary_intents(
    *,
    intents: List[PreferenceSummaryIntent],
    state: AgentState,
    max_intents: int,
) -> List[PreferenceSummaryIntent]:
    allowed_objects = set(state["seen_objects"])
    existing_hypotheses = _existing_preference_texts(state)
    rejected_hypotheses = {item.strip().lower() for item in state["rejected_hypotheses"] if item.strip()}

    normalized: List[PreferenceSummaryIntent] = []
    seen_signatures = set()

    for item in intents:
        hypothesis = item.hypothesis.strip()
        question = item.question.strip()

        if not hypothesis or not question:
            continue
        if hypothesis.lower() in existing_hypotheses:
            continue
        if hypothesis.lower() in rejected_hypotheses:
            continue

        covered_objects = [
            obj for obj in _dedupe_keep_order(list(item.covered_objects))
            if obj in allowed_objects
        ]
        if len(covered_objects) < 2:
            continue

        signature = (
            hypothesis.lower(),
            tuple(sorted(covered_objects)),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        normalized.append(
            PreferenceSummaryIntent(
                question_pattern="preference_summary",
                hypothesis=hypothesis,
                covered_objects=covered_objects,
                priority=_clip_priority(item.priority),
                question=question,
            )
        )

    normalized.sort(key=lambda x: x.priority, reverse=True)
    return normalized[:max_intents]


# =========================================================
# Convenience wrappers
# =========================================================

def propose_preference_eliciting_intents(
    *,
    state: AgentState,
    proposer: PreferenceElicitingProposer,
    max_intents: int = 4,
) -> List[PreferenceElicitingIntent]:
    return proposer.propose(state=state, max_intents=max_intents)


def propose_action_intent(
    *,
    state: AgentState,
    proposer: ActionProposer,
) -> Optional[ActionIntent]:
    return proposer.propose(state=state)


def propose_preference_summary_intents(
    *,
    state: AgentState,
    proposer: PreferenceSummaryProposer,
    max_intents: int = 3,
) -> List[PreferenceSummaryIntent]:
    intents = proposer.propose(state=state, max_intents=max_intents)
    for intent in intents:
        _upsert_preference_candidate(
            state=state,
            hypothesis=intent.hypothesis,
            covered_objects=list(intent.covered_objects),
        )
    return intents


# =========================================================
# CLI / smoke test
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to dataset JSON file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Episode index.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="parallel_exploration",
        choices=["direct", "preference_first", "parallel_exploration"],
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["eliciting", "action", "summary", "all"],
        help="Which proposer(s) to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=QUESTION_MODEL,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=OLLAMA_BASE_URL,
    )
    args = parser.parse_args()

    episode = get_episode(args.data, index=args.index)
    state = build_initial_state(
        episode=episode,
        strategy=args.strategy,  # type: ignore[arg-type]
        budget_total=args.budget,
    )
    updater = StateUpdate(model=args.model, base_url=args.base_url, temperature=0.0)
    updater.update_open_preference_hypotheses(state=state)

    print("=== Current State ===")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print()

    eliciting_proposer = PreferenceElicitingProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )
    action_proposer = ActionProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )
    summary_proposer = PreferenceSummaryProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )

    if args.mode in ("eliciting", "all"):
        print("=== Preference-eliciting proposer ===")
        intents = propose_preference_eliciting_intents(
            state=state,
            proposer=eliciting_proposer,
            max_intents=4,
        )
        for intent in intents:
            print(intent.model_dump())
        print()

    if args.mode in ("action", "all"):
        print("=== Action-oriented proposer ===")
        intent = propose_action_intent(
            state=state,
            proposer=action_proposer,
        )
        print(intent.model_dump() if intent is not None else None)
        print()

    if args.mode in ("summary", "all"):
        print("=== Preference-summary proposer ===")
        # Mock a little evidence so summary proposer has something to infer from
        state["confirmed_actions"] = {
            state["seen_objects"][0]: state["receptacles"][0],
            state["seen_objects"][1]: state["receptacles"][0],
        }
        print("=== Current State For Preference-summary ===")
        print(json.dumps(state, indent=2, ensure_ascii=False))
        print()
        intents = propose_preference_summary_intents(
            state=state,
            proposer=summary_proposer,
            max_intents=3,
        )
        for intent in intents:
            print(intent.model_dump())
        print()


if __name__ == "__main__":
    main()
