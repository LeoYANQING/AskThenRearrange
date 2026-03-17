from __future__ import annotations

import argparse
from typing import List

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from agent_schema import AgentState, Episode, Strategy


# =========================
# Config: LLMs (Ollama)
# =========================
BOOTSTRAP_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


# We keep the candidate dimension space intentionally small and stable.
ALLOWED_PREFERENCE_DIMENSIONS = [
    "fragility",
    "accessibility",
    "cleaning_food_separation",
    "heaviness",
    "activity_grouping",
    "prep_tool_separation",
]


class OpenPreferenceDimensionsOutput(BaseModel):
    dimensions: List[str] = Field(
        description=(
            "A small, conservative list of high-level preference dimensions "
            "worth directly asking about in the current visible scene."
        )
    )


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _fallback_bootstrap_open_preference_dimensions(episode: Episode) -> List[str]:
    """
    Minimal fallback if Ollama is unavailable.
    This remains intentionally lightweight and only uses seen_objects.
    """
    objects = " ".join(x.lower() for x in episode.seen_objects)
    dims: List[str] = []

    if any(k in objects for k in ["glass", "cup", "mug", "wine glass", "bowl"]):
        dims.append("fragility")

    if any(k in objects for k in ["salt", "pepper", "oil", "vinegar", "soy sauce", "paper towel"]):
        dims.append("accessibility")

    if any(k in objects for k in ["soap", "sponge", "bleach", "cleaner"]):
        dims.append("cleaning_food_separation")

    if any(k in objects for k in ["pan", "pot", "lid", "cast iron"]):
        dims.append("heaviness")

    if any(k in objects for k in ["coffee", "tea", "filter"]):
        dims.append("activity_grouping")

    if any(k in objects for k in ["cutting board", "knife", "paring", "chef", "meat", "fruit"]):
        dims.append("prep_tool_separation")

    return _dedupe_keep_order([d for d in dims if d in ALLOWED_PREFERENCE_DIMENSIONS])


def bootstrap_open_preference_dimensions_llm(
    episode: Episode,
    *,
    model: str = BOOTSTRAP_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.0,
) -> List[str]:
    """
    LLM-based initialization of open_preference_dimensions.

    Uses ONLY:
    - room
    - receptacles
    - seen_objects

    Does NOT use:
    - unseen_objects
    - annotator_notes
    - gt_seen_placements
    - gt_unseen_placements
    """
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    ).with_structured_output(OpenPreferenceDimensionsOutput)

    system_prompt = f"""
You are initializing the belief state for a household rearrangement agent.

Your task:
Given the visible scene only, identify which high-level preference dimensions
are worth directly asking the user about.

Important:
- You are NOT inferring the user's true preferences.
- You are only deciding which dimensions are worth asking about.
- Be conservative and return only a small set of likely useful dimensions.
- Use ONLY the visible scene information.
- Do NOT use hidden or oracle-only information.
- The output dimensions must come only from this allowed set:

{ALLOWED_PREFERENCE_DIMENSIONS}
""".strip()

    user_prompt = f"""
Visible scene information:

room:
{episode.room}

receptacles:
{episode.receptacles}

seen_objects:
{episode.seen_objects}

Return a small list of preference dimensions that are worth directly asking about.
""".strip()

    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    dims = [d for d in result.dimensions if d in ALLOWED_PREFERENCE_DIMENSIONS]
    return _dedupe_keep_order(dims)


def bootstrap_open_preference_dimensions(
    episode: Episode,
    *,
    use_llm_bootstrap: bool = True,
    model: str = BOOTSTRAP_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.0,
) -> List[str]:
    """
    Wrapper with a lightweight fallback.

    Default:
    - try LLM bootstrap
    - if it fails, fall back to a minimal heuristic bootstrap
    """
    if not use_llm_bootstrap:
        return _fallback_bootstrap_open_preference_dimensions(episode)

    try:
        return bootstrap_open_preference_dimensions_llm(
            episode,
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
    except Exception as e:
        print(f"[state_init] LLM bootstrap failed, using fallback. Error: {e}")
        return _fallback_bootstrap_open_preference_dimensions(episode)


def build_initial_state(
    episode: Episode,
    *,
    strategy: Strategy,
    budget_total: int,
    use_llm_bootstrap: bool = True,
    bootstrap_model: str = BOOTSTRAP_MODEL,
    bootstrap_base_url: str = OLLAMA_BASE_URL,
    bootstrap_temperature: float = 0.0,
) -> AgentState:
    """
    Build the minimal explicit state for one episode.

    Semantic roles:
    - open_preference_dimensions:
        unknown but worth directly asking
    - preference_candidates:
        induced but not yet confirmed rules
    - confirmed_preferences:
        confirmed user preference rules
    - confirmed_actions:
        confirmed seen object -> receptacle evidence
    - unresolved_objects:
        seen objects whose placements are not yet grounded
    """
    open_dims = bootstrap_open_preference_dimensions(
        episode,
        use_llm_bootstrap=use_llm_bootstrap,
        model=bootstrap_model,
        base_url=bootstrap_base_url,
        temperature=bootstrap_temperature,
    )

    state: AgentState = {
        # control
        "strategy": strategy,
        "budget_total": budget_total,
        "budget_used": 0,

        # task input
        "room": episode.room,
        "receptacles": list(episode.receptacles),
        "seen_objects": list(episode.seen_objects),
        "unseen_objects": list(episode.unseen_objects),
        "seen_placements": dict(episode.seen_placements),
        "unseen_placements": dict(episode.unseen_placements),
        "annotator_notes": list(episode.annotator_notes),

        # interaction evidence
        "qa_history": [],
        "asked_questions": [],
        "confirmed_actions": {},

        # preference belief state
        "open_preference_dimensions": open_dims,
        "preference_candidates": [],
        "confirmed_preferences": [],

        # object solving state
        "unresolved_objects": list(episode.seen_objects),

        # transient fields
        "current_pattern": None,
        "current_question": None,
        "current_answer": None,

        # outputs
        "predicted_placements_seen": {},
        "predicted_placements_unseen": {},
    }

    return state


def _print_state_summary(state: AgentState) -> None:
    print("strategy:", state["strategy"])
    print("budget_total:", state["budget_total"])
    print("budget_used:", state["budget_used"])
    print("num_seen_objects:", len(state["seen_objects"]))
    print("num_unresolved_objects:", len(state["unresolved_objects"]))
    print("num_unseen_objects:", len(state["unseen_objects"]))
    print("open_preference_dimensions:", state["open_preference_dimensions"])
    print("confirmed_actions:", state["confirmed_actions"])
    print("preference_candidates:", state["preference_candidates"])
    print("confirmed_preferences:", state["confirmed_preferences"])


if __name__ == "__main__":
    from data import get_episode

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="v1/data/scenarios_aug_tiny.json",
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
        "--no-llm-bootstrap",
        action="store_true",
        help="Disable LLM bootstrap and use fallback bootstrap instead.",
    )
    parser.add_argument(
        "--bootstrap-model",
        type=str,
        default=BOOTSTRAP_MODEL,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=OLLAMA_BASE_URL,
    )
    args = parser.parse_args()

    episode = get_episode(args.data, index=args.index)

    state = build_initial_state(
        episode,
        strategy=args.strategy,  # type: ignore[arg-type]
        budget_total=args.budget,
        use_llm_bootstrap=not args.no_llm_bootstrap,
        bootstrap_model=args.bootstrap_model,
        bootstrap_base_url=args.base_url,
        bootstrap_temperature=0.0,
    )

    print("=== Initial State Smoke Test ===")
    _print_state_summary(state)
    print()

    print("=== LLM Bootstrap Check ===")
    print("room:", state["room"])
    print("seen_objects_first5:", state["seen_objects"][:5])
    print("open_preference_dimensions:", state["open_preference_dimensions"])
    print()

    print("=== Sanity Checks ===")
    unseen_in_unresolved = any(
        obj in state["unresolved_objects"] for obj in state["unseen_objects"]
    )
    print("unseen_objects appear in unresolved_objects? ->", unseen_in_unresolved)
    print("confirmed_actions empty? ->", state["confirmed_actions"] == {})
    print("preference_candidates empty? ->", state["preference_candidates"] == [])
    print("confirmed_preferences empty? ->", state["confirmed_preferences"] == [])