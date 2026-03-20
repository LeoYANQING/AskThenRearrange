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
        PreferenceElicitingIntent,
        PreferenceElicitingIntentBatch,
        PreferenceSummaryIntent,
        PreferenceSummaryIntentBatch,
    )
    from v2.data import get_episode
    from v2.state_init import build_initial_state
    from v2.state_update import StateUpdate, update_open_preference_dimensions
except ModuleNotFoundError:
    from agent_schema import (
        ActionIntent,
        AgentState,
        PreferenceElicitingIntent,
        PreferenceElicitingIntentBatch,
        PreferenceSummaryIntent,
        PreferenceSummaryIntentBatch,
    )
    from data import get_episode
    from state_init import build_initial_state
    from state_update import StateUpdate, update_open_preference_dimensions


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
    return existing


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
        )
        self.structured_model = self.model.with_structured_output(
            PreferenceElicitingIntentBatch
        )

    def propose(
        self,
        *,
        state: AgentState,
        max_intents: int = 4,
    ) -> List[PreferenceElicitingIntent]:
        system_prompt = f"""
You are a proposer for Preference-eliciting questions in a household rearrangement task.

Your job:
Given the visible scene and current state, propose a small number of high-level
preference dimensions that are worth asking the user directly.

Important:
- This is an empirical pattern called "preference_eliciting".
- Do NOT output action-oriented questions.
- Do NOT output preference-summary questions.
- Be conservative.
- Return at most {max_intents} intents.
- Each intent must already include a natural user-facing question.
- Use only exact seen object names in covered_objects.

You may use:
- room
- receptacles
- seen_objects
- open_preference_dimensions
- already confirmed preferences

You must not use:
- unseen_objects
- hidden information
""".strip()

        user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Open preference dimensions:
{state["open_preference_dimensions"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Return a small list of Preference-eliciting intents.

Use the current open_preference_dimensions as the primary source of candidate dimensions.
If that list is non-empty, prefer proposing only from that list unless a listed dimension is clearly no longer useful.

Each intent must:
- source = "scene_gap"
- question_pattern = "preference_eliciting"
- dimension = a concise missing high-level preference dimension
- covered_objects = exact seen objects related to that dimension
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
    open_dimensions = {item.strip().lower() for item in state["open_preference_dimensions"] if item.strip()}
    normalized: List[PreferenceElicitingIntent] = []
    seen_signatures = set()

    for item in intents:
        dimension = item.dimension.strip()
        question = item.question.strip()
        if not dimension or not question:
            continue
        if open_dimensions and dimension.lower() not in open_dimensions:
            continue

        covered_objects = [
            obj for obj in _dedupe_keep_order(list(item.covered_objects))
            if obj in allowed_objects
        ]
        if not covered_objects:
            continue

        signature = (
            dimension.lower(),
            tuple(sorted(covered_objects)),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        normalized.append(
            PreferenceElicitingIntent(
                source="scene_gap",
                question_pattern="preference_eliciting",
                dimension=dimension,
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
        )
        self.structured_model = self.model.with_structured_output(ActionIntent)

    def propose(
        self,
        *,
        state: AgentState,
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
- Be conservative and stable.
- Prefer boundary_probe only when there is a plausible confirmed preference whose boundary can be tested with the object.
""".strip()

        user_prompt = f"""
Unresolved seen objects:
{state["unresolved_objects"]}

Confirmed actions:
{state["confirmed_actions"]}

Preference candidates:
{state["preference_candidates"]}

Confirmed preferences:
{state["confirmed_preferences"]}

QA history:
{state["qa_history"]}

Return exactly one ActionIntent:
- source = "action"
- question_pattern = "action_oriented"
- action_mode = "direct_grounding" or "boundary_probe"
- object_name = one exact unresolved seen object
- priority = 0.0 to 1.0
- question = one concise natural Action-oriented question
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
        source="action",
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
        )
        self.structured_model = self.model.with_structured_output(
            PreferenceSummaryIntentBatch
        )

    def propose(
        self,
        *,
        state: AgentState,
        max_intents: int = 3,
    ) -> List[PreferenceSummaryIntent]:
        if len(state["confirmed_actions"]) < 2:
            return []

        system_prompt = f"""
You are a proposer for Preference-summary questions in a household rearrangement task.

Your job:
Infer a small number of high-level preference summaries from the confirmed action evidence,
then generate concise summary-style questions asking the user to confirm or refine them.

Important:
- This empirical pattern is "preference_summary".
- Do NOT output action-oriented questions.
- Do NOT output preference-eliciting questions.
- Be conservative.
- Return at most {max_intents} intents.
- Do not repeat an already existing candidate or confirmed preference.
- Each intent must already include a natural user-facing summary question.
- Use only exact seen object names in covered_objects.
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

Return a small list of Preference-summary intents.

Each intent must:
- source = "induced_hypothesis"
- question_pattern = "preference_summary"
- hypothesis = a concise rule summary to confirm
- covered_objects = exact seen objects plausibly covered by the summary
- priority = 0.0 to 1.0
- question = one concise natural summary question asking for confirmation or correction
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

    normalized: List[PreferenceSummaryIntent] = []
    seen_signatures = set()

    for item in intents:
        hypothesis = item.hypothesis.strip()
        question = item.question.strip()

        if not hypothesis or not question:
            continue
        if hypothesis.lower() in existing_hypotheses:
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
                source="induced_hypothesis",
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
    return proposer.propose(state=state, max_intents=max_intents)


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
    state = update_open_preference_dimensions(
        state=state,
        updater=StateUpdate(model=args.model, base_url=args.base_url, temperature=0.0),
    )

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
