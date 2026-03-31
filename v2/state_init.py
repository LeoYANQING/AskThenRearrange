from __future__ import annotations

import argparse

try:
    from v2.agent_schema import AgentState, Episode, Strategy
except ModuleNotFoundError:
    from agent_schema import AgentState, Episode, Strategy


def build_initial_state(
    episode: Episode,
    *,
    strategy: Strategy,
    budget_total: int,
) -> AgentState:
    """
    Build the minimal explicit state for one episode.

    Semantic roles:
    - confirmed_actions:
        confirmed seen object -> receptacle evidence
    - unresolved_objects:
        seen objects whose placements are not yet grounded
    """
    state: AgentState = {
        "budget_total": budget_total,

        # task input
        "room": episode.room,
        "receptacles": list(episode.receptacles),
        "seen_objects": list(episode.seen_objects),
        "unseen_objects": list(episode.unseen_objects),

        # interaction evidence
        "qa_history": [],
        "confirmed_actions": [],

        # preference belief state
        "confirmed_preferences": [],
        "negative_preferences": [],
        "negative_actions": [],

        # object solving state
        "unresolved_objects": list(episode.seen_objects),
    }

    return state


def _print_state_summary(state: AgentState) -> None:
    print("budget_total:", state["budget_total"])
    print("budget_used:", len(state["qa_history"]))
    print("num_seen_objects:", len(state["seen_objects"]))
    print("num_unresolved_objects:", len(state["unresolved_objects"]))
    print("num_unseen_objects:", len(state["unseen_objects"]))
    print("confirmed_preferences:", state["confirmed_preferences"])
    print("confirmed_actions:", state["confirmed_actions"])
    print("negative_preferences:", state["negative_preferences"])
    print("negative_actions:", state["negative_actions"])


if __name__ == "__main__":
    try:
        from v2.data import get_episode
    except ModuleNotFoundError:
        from data import get_episode

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="v2/data/scenarios_aug_tiny.json",
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
    args = parser.parse_args()

    episode = get_episode(args.data, index=args.index)

    state = build_initial_state(
        episode,
        strategy=args.strategy,  # type: ignore[arg-type]
        budget_total=args.budget,
    )

    print("=== Initial State Smoke Test ===")
    _print_state_summary(state)
    print()

    print("=== Sanity Checks ===")
    unseen_in_unresolved = any(
        obj in state["unresolved_objects"] for obj in state["unseen_objects"]
    )
    print("unseen_objects appear in unresolved_objects? ->", unseen_in_unresolved)
    print("confirmed_actions empty? ->", state["confirmed_actions"] == [])
    print("confirmed_preferences empty? ->", state["confirmed_preferences"] == [])
