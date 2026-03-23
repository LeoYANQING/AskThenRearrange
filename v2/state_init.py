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
    - preference_candidates:
        induced but not yet confirmed rules
    - confirmed_preferences:
        confirmed user preference rules
    - confirmed_actions:
        confirmed seen object -> receptacle evidence
    - unresolved_objects:
        seen objects whose placements are not yet grounded
    """
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

        # interaction evidence
        "qa_history": [],
        "confirmed_actions": {},
        "excluded_receptacles": {},

        # preference belief state
        "open_preference_hypotheses": [],
        "preference_candidates": [],
        "confirmed_preferences": [],
        "rejected_hypotheses": [],

        # object solving state
        "unresolved_objects": list(episode.seen_objects),

        # online derived placements
        "online_placements_seen": {},
    }

    return state


def _print_state_summary(state: AgentState) -> None:
    print("strategy:", state["strategy"])
    print("budget_total:", state["budget_total"])
    print("budget_used:", state["budget_used"])
    print("num_seen_objects:", len(state["seen_objects"]))
    print("num_unresolved_objects:", len(state["unresolved_objects"]))
    print("num_unseen_objects:", len(state["unseen_objects"]))
    print("open_preference_hypotheses:", state["open_preference_hypotheses"])
    print("confirmed_actions:", state["confirmed_actions"])
    print("preference_candidates:", state["preference_candidates"])
    print("confirmed_preferences:", state["confirmed_preferences"])


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
    print("confirmed_actions empty? ->", state["confirmed_actions"] == {})
    print("preference_candidates empty? ->", state["preference_candidates"] == [])
    print("confirmed_preferences empty? ->", state["confirmed_preferences"] == [])
