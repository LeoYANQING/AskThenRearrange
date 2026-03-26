from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from v2.agent_schema import AgentState
    from v2.data import DEFAULT_DATA_PATH, Episode, get_episode
    from v2.evaluation import FinalPlacementPlanner, evaluate_episode_state, plot_accuracy_curve
    from v2.oracle import NaturalUserOracle
    from v2.proposers import PreferenceElicitingProposer
    from v2.state_init import build_initial_state
    from v2.state_update import StateUpdate
except ModuleNotFoundError:
    from agent_schema import AgentState
    from data import DEFAULT_DATA_PATH, Episode, get_episode
    from evaluation import FinalPlacementPlanner, evaluate_episode_state, plot_accuracy_curve
    from oracle import NaturalUserOracle
    from proposers import PreferenceElicitingProposer
    from state_init import build_initial_state
    from state_update import StateUpdate


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _state_snapshot(state: AgentState) -> Dict[str, Any]:
    return {
        "budget_used": state["budget_used"],
        "open_preference_hypotheses": state["open_preference_hypotheses"],
        "confirmed_actions": state["confirmed_actions"],
        "preference_candidates": state["preference_candidates"],
        "confirmed_preferences": state["confirmed_preferences"],
        "rejected_hypotheses": state["rejected_hypotheses"],
        "online_placements_seen": state["online_placements_seen"],
                "unresolved_objects": state["unresolved_objects"],
        "qa_history_len": len(state["qa_history"]),
    }


def _parse_budget_list(value: str) -> List[int]:
    budgets: List[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        budget = int(stripped)
        if budget <= 0:
            raise ValueError(f"budget values must be positive, got {budget}")
        budgets.append(budget)
    if not budgets:
        raise ValueError("budget list must not be empty")
    return budgets


def run_preference_eliciting_loop(
    *,
    episode: Episode,
    state: AgentState,
    proposer: PreferenceElicitingProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
) -> AgentState:
    step_idx = 0
    while state["budget_used"] < state["budget_total"]:
        step_idx += 1

        print(f"[step {step_idx}] proposing preference-eliciting intents...")
        intents = proposer.propose(state=state, max_intents=3)
        if not intents:
            print(f"[step {step_idx}] no preference-eliciting intent available")
            break

        intent = intents[0]
        print(f"[step {step_idx}] selected hypothesis: {intent.hypothesis}")
        print(f"[step {step_idx}] querying oracle...")

        oracle_response = oracle.answer(
            question=intent.question,
            room=state["room"],
            receptacles=state["receptacles"],
            seen_objects=state["seen_objects"],
            annotator_notes=episode.annotator_notes,
            gt_seen_placements=episode.seen_placements,
            qa_history=state["qa_history"],
        )

        print(f"[step {step_idx}] updating state from answer...")
        state = updater.update_state_from_preference_eliciting_answer(
            state=state,
            hypothesis=intent.hypothesis,
            covered_objects=list(intent.covered_objects),
            answer=oracle_response.answer,
            question=intent.question,
        )
        print(f"[step {step_idx}] state update finished")

        print(f"=== Step {step_idx} ===")
        print(
            json.dumps(
                {   "question": intent.question,
                    "oracle_response": oracle_response.model_dump(),
                    "state": _state_snapshot(state),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print()

    return state


def run_preference_eliciting_episode(
    *,
    episode: Episode,
    budget: int,
    proposer_model: str,
    oracle_model: str,
    updater_model: str,
    evaluation_model: str,
    base_url: str,
    verbose: bool,
) -> tuple[AgentState, Dict[str, Any]]:
    state = build_initial_state(
        episode=episode,
        strategy="parallel_exploration",
        budget_total=budget,
    )
    proposer = PreferenceElicitingProposer(
        model=proposer_model,
        base_url=base_url,
        temperature=0.0,
    )
    oracle = NaturalUserOracle(
        model=oracle_model,
        base_url=base_url,
        temperature=0.0,
    )
    updater = StateUpdate(
        model=updater_model,
        base_url=base_url,
        temperature=0.0,
    )
    planner = FinalPlacementPlanner(
        model=evaluation_model,
        base_url=base_url,
        temperature=0.0,
    )
    updater.update_open_preference_hypotheses(state=state)

    if verbose:
        print("=== Initial State ===")
        print(json.dumps(_state_snapshot(state), indent=2, ensure_ascii=False))
        print()

    final_state = run_preference_eliciting_loop(
        episode=episode,
        state=state,
        proposer=proposer,
        oracle=oracle,
        updater=updater,
    )
    evaluation = evaluate_episode_state(
        episode,
        final_state,
        planner=planner,
    )
    return final_state, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for PreferenceElicitingProposer + Oracle + StateUpdate."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--proposer-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--oracle-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--updater-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--evaluation-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    parser.add_argument("--plot-curve", action="store_true", default=True)
    parser.add_argument("--curve-output", type=str, default="")
    parser.add_argument("--budget-list", type=str, default="5")
    args = parser.parse_args()

    episode = get_episode(Path(args.data), args.index)
    budgets = _parse_budget_list(args.budget_list)
    if args.plot_curve:
        curve_points: List[Dict[str, Any]] = []
        for budget in budgets:
            _, evaluation = run_preference_eliciting_episode(
                episode=episode,
                budget=budget,
                proposer_model=args.proposer_model,
                oracle_model=args.oracle_model,
                updater_model=args.updater_model,
                evaluation_model=args.evaluation_model,
                base_url=args.base_url,
                verbose=False,
            )
            curve_points.append(
                {
                    "budget": budget,
                    "seen_accuracy": evaluation["seen_accuracy"],
                    "unseen_accuracy": evaluation["unseen_accuracy"],
                }
            )

        output_path = args.curve_output or "v2/plots/preference_eliciting_loop_accuracy_curve.png"
        saved_path = plot_accuracy_curve(
            curve_points,
            output_path=output_path,
            title=f"Preference-Eliciting Loop Accuracy vs Budget ({episode.episode_id})",
        )
        print(json.dumps({"curve_points": curve_points, "saved_plot": saved_path}, indent=2, ensure_ascii=False))
        return

    for budget in budgets:
        final_state, evaluation = run_preference_eliciting_episode(
            episode=episode,
            budget=budget,
            proposer_model=args.proposer_model,
            oracle_model=args.oracle_model,
            updater_model=args.updater_model,
            evaluation_model=args.evaluation_model,
            base_url=args.base_url,
            verbose=True,
        )

        print(f"=== Budget {budget} Final State ===")
        print(json.dumps(final_state, indent=2, ensure_ascii=False))
        print()

        print(f"=== Budget {budget} Evaluation ===")
        print(json.dumps(evaluation, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
