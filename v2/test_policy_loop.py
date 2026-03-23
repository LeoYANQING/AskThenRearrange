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
    from v2.proposers import ActionProposer, PreferenceElicitingProposer, PreferenceSummaryProposer
    from v2.question_policy import PolicyMode, QuestionDecision, QuestionPolicyController
    from v2.state_init import build_initial_state
    from v2.state_update import StateUpdate
except ModuleNotFoundError:
    from agent_schema import AgentState
    from data import DEFAULT_DATA_PATH, Episode, get_episode
    from evaluation import FinalPlacementPlanner, evaluate_episode_state, plot_accuracy_curve
    from oracle import NaturalUserOracle
    from proposers import ActionProposer, PreferenceElicitingProposer, PreferenceSummaryProposer
    from question_policy import PolicyMode, QuestionDecision, QuestionPolicyController
    from state_init import build_initial_state
    from state_update import StateUpdate


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _state_snapshot(state: AgentState) -> Dict[str, Any]:
    return {
        "budget_used": state["budget_used"],
        "open_preference_hypotheses": state["open_preference_hypotheses"],
        "confirmed_actions": state["confirmed_actions"],
        "excluded_receptacles": state["excluded_receptacles"],
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


def _propose_intent(
    *,
    state: AgentState,
    decision: QuestionDecision,
    eliciting_proposer: PreferenceElicitingProposer,
    action_proposer: ActionProposer,
    summary_proposer: PreferenceSummaryProposer,
):
    if decision.question_pattern == "preference_eliciting":
        intents = eliciting_proposer.propose(
            state=state,
            max_intents=3,
            guidance=decision.guidance,
        )
        return intents[0] if intents else None

    if decision.question_pattern == "action_oriented":
        return action_proposer.propose(
            state=state,
            guidance=decision.guidance,
        )

    if decision.question_pattern == "preference_summary":
        intents = summary_proposer.propose(
            state=state,
            max_intents=3,
            guidance=decision.guidance,
        )
        return intents[0] if intents else None

    raise ValueError(f"Unsupported question pattern: {decision.question_pattern}")


def run_policy_loop(
    *,
    episode: Episode,
    state: AgentState,
    mode: PolicyMode,
    controller: QuestionPolicyController,
    eliciting_proposer: PreferenceElicitingProposer,
    action_proposer: ActionProposer,
    summary_proposer: PreferenceSummaryProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
) -> AgentState:
    step_idx = 0
    while state["budget_used"] < state["budget_total"]:
        step_idx += 1
        print(f"[step {step_idx}] planning policy...")
        decision = controller.plan_next_question(
            state=state,
            mode=mode,
        )
        if decision is None:
            print(f"[step {step_idx}] no policy decision available")
            break

        print(f"[step {step_idx}] proposing intent for {decision.question_pattern}...")
        intent = _propose_intent(
            state=state,
            decision=decision,
            eliciting_proposer=eliciting_proposer,
            action_proposer=action_proposer,
            summary_proposer=summary_proposer,
        )
        if intent is None:
            print(f"[step {step_idx}] no proposer intent available for {decision.question_pattern}")
            break

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

        print(f"[step {step_idx}] updating state...")
        if decision.question_pattern == "preference_eliciting":
            state = updater.update_state_from_preference_eliciting_answer(
                state=state,
                hypothesis=intent.hypothesis,
                covered_objects=list(intent.covered_objects),
                answer=oracle_response.answer,
                question=intent.question,
            )
        elif decision.question_pattern == "action_oriented":
            state = updater.update_state_from_action_answer(
                state=state,
                target=intent.object_name,
                answer=oracle_response.answer,
                question=intent.question,
                action_mode=intent.action_mode,
            )
        elif decision.question_pattern == "preference_summary":
            state = updater.update_state_from_preference_summary_answer(
                state=state,
                hypothesis=intent.hypothesis,
                covered_objects=list(intent.covered_objects),
                answer=oracle_response.answer,
                question=intent.question,
            )
        else:
            raise ValueError(f"Unsupported pattern: {decision.question_pattern}")

        print(f"=== Step {step_idx} ===")
        print(
            json.dumps(
                {
                    "policy_mode": mode,
                    "question_pattern": decision.question_pattern,
                    "guidance": decision.guidance,
                    "intent": intent.model_dump(),
                    "oracle_response": oracle_response.model_dump(),
                    "state": _state_snapshot(state),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print()

    return state


def run_policy_episode(
    *,
    episode: Episode,
    budget: int,
    mode: PolicyMode,
    proposer_model: str,
    oracle_model: str,
    updater_model: str,
    evaluation_model: str,
    base_url: str,
    verbose: bool,
) -> tuple[AgentState, Dict[str, Any]]:
    state = build_initial_state(
        episode=episode,
        strategy="",
        budget_total=budget,
    )
    controller = QuestionPolicyController(
        model=proposer_model,
        base_url=base_url,
        temperature=0.0,
    )
    eliciting_proposer = PreferenceElicitingProposer(
        model=proposer_model,
        base_url=base_url,
        temperature=0.0,
    )
    action_proposer = ActionProposer(
        model=proposer_model,
        base_url=base_url,
        temperature=0.0,
    )
    summary_proposer = PreferenceSummaryProposer(
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

    final_state = run_policy_loop(
        episode=episode,
        state=state,
        mode=mode,
        controller=controller,
        eliciting_proposer=eliciting_proposer,
        action_proposer=action_proposer,
        summary_proposer=summary_proposer,
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
    parser = argparse.ArgumentParser(description="Policy-driven multi-pattern dialogue loop.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        default="user_preference_first",
        choices=["direct_querying", "user_preference_first", "parallel_exploration", "hybrid_all"],
    )
    parser.add_argument("--proposer-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--oracle-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--updater-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--evaluation-model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    parser.add_argument("--plot-curve", action="store_true", default=False)
    parser.add_argument("--curve-output", type=str, default="")
    parser.add_argument("--budget-list", type=str, default="5")
    args = parser.parse_args()

    episode = get_episode(Path(args.data), args.index)
    budgets = _parse_budget_list(args.budget_list)

    if args.plot_curve:
        curve_points: List[Dict[str, Any]] = []
        for budget in budgets:
            _, evaluation = run_policy_episode(
                episode=episode,
                budget=budget,
                mode=args.mode,
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

        output_path = args.curve_output or f"v2/plots/{args.mode}_policy_accuracy_curve.png"
        saved_path = plot_accuracy_curve(
            curve_points,
            output_path=output_path,
            title=f"{args.mode} Accuracy vs Budget ({episode.episode_id})",
        )
        print(json.dumps({"curve_points": curve_points, "saved_plot": saved_path}, indent=2, ensure_ascii=False))
        return

    for budget in budgets:
        final_state, evaluation = run_policy_episode(
            episode=episode,
            budget=budget,
            mode=args.mode,
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
