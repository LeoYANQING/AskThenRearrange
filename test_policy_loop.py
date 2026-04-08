from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from agent_schema import AgentState
from data import DEFAULT_DATA_PATH, Episode, get_episode
from evaluation import FinalPlacementPlanner, evaluate_episode_state, plot_ablation_comparison, plot_accuracy_curve
from oracle import NaturalUserOracle
from proposers import ActionProposer, PreferenceElicitingProposer, PreferenceInductionProposer
from question_policy import PolicyMode, QuestionDecision, QuestionPolicyController, SelectionMethod
from state_init import build_initial_state
from state_update import StateUpdate


from llm_factory import DEFAULT_MODEL, DEFAULT_BASE_URL
QUESTION_MODEL = DEFAULT_MODEL
OLLAMA_BASE_URL = DEFAULT_BASE_URL


POLICY_LABELS = {
    "direct_querying": "Direct Querying",
    "user_preference_first": "User-Preference-First",
    "parallel_exploration": "Parallel Exploration",
    "hybrid_all": "Hybrid-All",
}


def _state_snapshot(state: AgentState) -> Dict[str, Any]:
    return {
        "budget_used": len(state["qa_history"]),
        "confirmed_actions": state["confirmed_actions"],
        "negative_actions": state["negative_actions"],
        "confirmed_preferences": state["confirmed_preferences"],
        "negative_preferences": state["negative_preferences"],
        "unresolved_objects": state["unresolved_objects"],
        "qa_history_len": len(state["qa_history"]),
    }


def _print_policy_step(
    *,
    mode: PolicyMode,
    episode: Episode,
    step_idx: int,
    decision: QuestionDecision,
    intent: Any,
    oracle_response: Any,
    state: AgentState,
) -> None:
    question = intent.question if hasattr(intent, "question") else str(intent.get("question", ""))
    print(
        f"[Episode {episode.episode_id} | Question Step {step_idx}] "
        f"mode={mode} pattern={decision.question_pattern}"
    )
    print(f"  guidance: {decision.guidance}")
    print(f"  question: {question}")
    print(f"  answer: {oracle_response.answer}")
    snapshot = _state_snapshot(state)
    print(
        "  state:"
        f" budget_used={snapshot['budget_used']},"
        f" unresolved={len(snapshot['unresolved_objects'])},"
        f" confirmed_actions={len(snapshot['confirmed_actions'])},"
        f" confirmed_preferences={len(snapshot['confirmed_preferences'])},"
        f" negative_preferences={len(snapshot['negative_preferences'])},"
        f" negative_actions={len(snapshot['negative_actions'])}"
    )
    print(
        json.dumps(
            {
                "policy_mode": mode,
                "question_pattern": decision.question_pattern,
                "guidance": decision.guidance,
                "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent,
                "question": question,
                "oracle_response": oracle_response.model_dump(),
                "state": snapshot,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _select_sample_indices(*, num_samples: int, start_index: int = 0) -> List[int]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    return list(range(start_index, start_index + num_samples))


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



def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _propose_intent(
    *,
    state: AgentState,
    decision: QuestionDecision,
    eliciting_proposer: PreferenceElicitingProposer,
    action_proposer: ActionProposer,
    induction_proposer: PreferenceInductionProposer,
):
    if decision.question_pattern == "preference_eliciting":
        intent = eliciting_proposer.propose(
            state=state,
            guidance=decision.guidance,
        )
        return intent

    if decision.question_pattern == "action_oriented":
        return action_proposer.propose(
            state=state,
            guidance=decision.guidance,
        )

    if decision.question_pattern == "preference_induction":
        intents = induction_proposer.propose(
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
    induction_proposer: PreferenceInductionProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
) -> AgentState:
    step_idx = 0
    while len(state["qa_history"]) < state["budget_total"]:
        step_idx += 1
        decision = controller.plan_next_question(
            state=state,
            mode=mode,
        )
        if decision is None:
            print(f"[Episode {episode.episode_id} | Question Step {step_idx}] no policy decision available")
            break

        intent = _propose_intent(
            state=state,
            decision=decision,
            eliciting_proposer=eliciting_proposer,
            action_proposer=action_proposer,
            induction_proposer=induction_proposer,
        )
        if intent is None:
            if decision.question_pattern == "preference_induction":
                # PI proposer couldn't form a hypothesis; fall back to AO for this step
                print(f"[Episode {episode.episode_id} | Question Step {step_idx}] PI proposer returned None — falling back to action_oriented")
                decision = QuestionDecision(
                    question_pattern="action_oriented",
                    guidance="No induction pattern available; ask a direct placement question to gather more evidence.",
                )
                intent = _propose_intent(
                    state=state,
                    decision=decision,
                    eliciting_proposer=eliciting_proposer,
                    action_proposer=action_proposer,
                    induction_proposer=induction_proposer,
                )
            if intent is None:
                print(f"[Episode {episode.episode_id} | Question Step {step_idx}] no proposer intent available for {decision.question_pattern}")
                break

        question = intent.question if hasattr(intent, "question") else str(intent.get("question", ""))
        oracle_response = oracle.answer(
            question=question,
            room=state["room"],
            receptacles=state["receptacles"],
            seen_objects=state["seen_objects"],
            annotator_notes=episode.annotator_notes,
            gt_seen_placements=episode.seen_placements,
            qa_history=state["qa_history"],
        )

        if decision.question_pattern == "preference_eliciting":
            state = updater.update_state_from_preference_eliciting_answer(
                state=state,
                hypothesis=str(intent.get("hypothesis", "")),
                covered_objects=list(intent.get("covered_objects", [])),
                answer=oracle_response.answer,
                question=question,
                oracle_receptacle=oracle_response.referenced_receptacle,
            )
        elif decision.question_pattern == "action_oriented":
            state = updater.update_state_from_action_answer(
                state=state,
                target=intent.object_name,
                answer=oracle_response.answer,
                question=intent.question,
                action_mode=intent.action_mode,
            )
        elif decision.question_pattern == "preference_induction":
            state = updater.update_state_from_preference_induction_answer(
                state=state,
                hypothesis=str(intent.get("hypothesis", "")),
                covered_objects=list(intent.get("covered_objects", [])),
                answer=oracle_response.answer,
                question=question,
            )
        else:
            raise ValueError(f"Unsupported pattern: {decision.question_pattern}")

        _print_policy_step(
            mode=mode,
            episode=episode,
            step_idx=step_idx,
            decision=decision,
            intent=intent,
            oracle_response=oracle_response,
            state=state,
        )

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
    selection_method: SelectionMethod = "rule",
    eval_budgets: List[int] | None = None,
) -> tuple[AgentState, Dict[str, Any]] | tuple[AgentState, Dict[int, Dict[str, Any]]]:
    """Run a policy episode.

    If eval_budgets is provided, run up to max(eval_budgets) steps and evaluate
    at each budget checkpoint. Returns (final_state, {budget: evaluation}).
    Otherwise runs to `budget` and returns (final_state, evaluation) as before.
    """
    max_budget = max(eval_budgets) if eval_budgets else budget
    state = build_initial_state(
        episode=episode,
        strategy="parallel_exploration",
        budget_total=max_budget,
    )
    controller = QuestionPolicyController(
        model=proposer_model,
        base_url=base_url,
        temperature=0.0,
        selection_method=selection_method,
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
    induction_proposer = PreferenceInductionProposer(
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

    if verbose:
        print("=== Initial State ===")
        print(json.dumps(_state_snapshot(state), indent=2, ensure_ascii=False))
        print()

    if eval_budgets is not None:
        # Incremental mode: run step by step, evaluate at checkpoints
        eval_set = set(eval_budgets)
        results_by_budget: Dict[int, Dict[str, Any]] = {}
        step_idx = 0
        while len(state["qa_history"]) < max_budget:
            step_idx += 1
            decision = controller.plan_next_question(state=state, mode=mode)
            if decision is None:
                break
            intent = _propose_intent(
                state=state, decision=decision,
                eliciting_proposer=eliciting_proposer,
                action_proposer=action_proposer,
                induction_proposer=induction_proposer,
            )
            if intent is None:
                if decision.question_pattern == "preference_induction":
                    decision = QuestionDecision(
                        question_pattern="action_oriented",
                        guidance="No induction pattern available; ask a direct placement question.",
                    )
                    intent = _propose_intent(
                        state=state, decision=decision,
                        eliciting_proposer=eliciting_proposer,
                        action_proposer=action_proposer,
                        induction_proposer=induction_proposer,
                    )
                if intent is None:
                    break

            question = intent.question if hasattr(intent, "question") else str(intent.get("question", ""))
            oracle_response = oracle.answer(
                question=question, room=state["room"],
                receptacles=state["receptacles"],
                seen_objects=state["seen_objects"],
                annotator_notes=episode.annotator_notes,
                gt_seen_placements=episode.seen_placements,
                qa_history=state["qa_history"],
            )
            if decision.question_pattern == "preference_eliciting":
                state = updater.update_state_from_preference_eliciting_answer(
                    state=state, hypothesis=str(intent.get("hypothesis", "")),
                    covered_objects=list(intent.get("covered_objects", [])),
                    answer=oracle_response.answer, question=question,
                    oracle_receptacle=oracle_response.referenced_receptacle,
                )
            elif decision.question_pattern == "action_oriented":
                state = updater.update_state_from_action_answer(
                    state=state, target=intent.object_name,
                    answer=oracle_response.answer, question=intent.question,
                    action_mode=intent.action_mode,
                )
            elif decision.question_pattern == "preference_induction":
                state = updater.update_state_from_preference_induction_answer(
                    state=state, hypothesis=str(intent.get("hypothesis", "")),
                    covered_objects=list(intent.get("covered_objects", [])),
                    answer=oracle_response.answer, question=question,
                )

            _print_policy_step(
                mode=mode, episode=episode, step_idx=step_idx,
                decision=decision, intent=intent,
                oracle_response=oracle_response, state=state,
            )

            current_budget = len(state["qa_history"])
            if current_budget in eval_set:
                evaluation = evaluate_episode_state(episode, state, planner=planner)
                evaluation["qa_history"] = list(state["qa_history"])
                evaluation["confirmed_actions"] = list(state["confirmed_actions"])
                evaluation["confirmed_preferences"] = list(state["confirmed_preferences"])
                results_by_budget[current_budget] = evaluation

        # Fill any remaining eval_budgets that weren't reached (loop ended early)
        for b in eval_budgets:
            if b not in results_by_budget:
                evaluation = evaluate_episode_state(episode, state, planner=planner)
                evaluation["qa_history"] = list(state["qa_history"])
                evaluation["confirmed_actions"] = list(state["confirmed_actions"])
                evaluation["confirmed_preferences"] = list(state["confirmed_preferences"])
                results_by_budget[b] = evaluation

        return state, results_by_budget
    else:
        # Original mode: run to budget, evaluate once
        final_state = run_policy_loop(
            episode=episode, state=state, mode=mode,
            controller=controller,
            eliciting_proposer=eliciting_proposer,
            action_proposer=action_proposer,
            induction_proposer=induction_proposer,
            oracle=oracle, updater=updater,
        )
        evaluation = evaluate_episode_state(episode, final_state, planner=planner)
        return final_state, evaluation



def run_ablation_experiment(
    *,
    data_path: Path,
    sample_indices: List[int],
    budgets: List[int],
    proposer_model: str,
    oracle_model: str,
    updater_model: str,
    evaluation_model: str,
    base_url: str,
    log_path: Path | None = None,
    selection_method: SelectionMethod = "rule",
    modes: List[PolicyMode] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if modes is None:
        modes = [
            "direct_querying",
            "user_preference_first",
            "parallel_exploration",
            "hybrid_all",
        ]
    curves_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    experiment_started_at = time.perf_counter()

    if log_path is not None:
        _append_jsonl(
            log_path,
            {
                "event": "ablation_started",
                "sample_indices": sample_indices,
                "budgets": budgets,
                "modes": modes,
            },
        )

    total_runs = len(modes) * len(sample_indices)
    completed_runs = 0
    experiment_start = time.perf_counter()

    for mode_idx, mode in enumerate(modes):
        print(f"\n[Ablation] Mode {mode_idx+1}/{len(modes)}: {mode}", flush=True)
        budget_scores: Dict[int, Dict[str, List[float]]] = {
            b: {"seen": [], "unseen": [], "elapsed": []} for b in budgets
        }

        for ep_idx, index in enumerate(sample_indices):
            episode = get_episode(data_path, index)
            started_at = time.perf_counter()
            _, results_by_budget = run_policy_episode(
                episode=episode,
                budget=max(budgets),
                mode=mode,
                proposer_model=proposer_model,
                oracle_model=oracle_model,
                updater_model=updater_model,
                evaluation_model=evaluation_model,
                base_url=base_url,
                verbose=False,
                selection_method=selection_method,
                eval_budgets=budgets,
            )
            elapsed_sec = time.perf_counter() - started_at
            completed_runs += 1
            total_elapsed = time.perf_counter() - experiment_start
            eta = total_elapsed / completed_runs * (total_runs - completed_runs)
            if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
                print(f"  [{completed_runs}/{total_runs}] ep{index} {elapsed_sec:.0f}s | total {total_elapsed:.0f}s | ETA {eta:.0f}s ({eta/60:.0f}min)", flush=True)

            for b in budgets:
                evaluation = results_by_budget[b]
                budget_scores[b]["seen"].append(float(evaluation["seen_accuracy"]))
                budget_scores[b]["unseen"].append(float(evaluation["unseen_accuracy"]))
                budget_scores[b]["elapsed"].append(elapsed_sec / len(budgets))

                if log_path is not None:
                    _append_jsonl(
                        log_path,
                        {
                            "event": "episode_finished",
                            "mode": mode,
                            "budget": b,
                            "episode_index": index,
                            "episode_id": episode.episode_id,
                            "elapsed_sec": elapsed_sec,
                            "seen_accuracy": float(evaluation["seen_accuracy"]),
                            "unseen_accuracy": float(evaluation["unseen_accuracy"]),
                        },
                    )

        mode_points: List[Dict[str, Any]] = []
        for budget in budgets:
            seen_scores = budget_scores[budget]["seen"]
            unseen_scores = budget_scores[budget]["unseen"]
            episode_durations_sec = budget_scores[budget]["elapsed"]

            seen_mean = sum(seen_scores) / len(seen_scores)
            unseen_mean = sum(unseen_scores) / len(unseen_scores)
            if len(seen_scores) > 1:
                seen_var = sum((v - seen_mean) ** 2 for v in seen_scores) / (len(seen_scores) - 1)
                unseen_var = sum((v - unseen_mean) ** 2 for v in unseen_scores) / (len(unseen_scores) - 1)
                seen_stderr = math.sqrt(seen_var) / math.sqrt(len(seen_scores))
                unseen_stderr = math.sqrt(unseen_var) / math.sqrt(len(unseen_scores))
            else:
                seen_stderr = 0.0
                unseen_stderr = 0.0

            mean_episode_sec = sum(episode_durations_sec) / len(episode_durations_sec)

            point = {
                "budget": budget,
                "seen_accuracy": seen_mean,
                "unseen_accuracy": unseen_mean,
                "seen_stderr": seen_stderr,
                "unseen_stderr": unseen_stderr,
                "num_episodes": len(sample_indices),
                "mean_episode_sec": mean_episode_sec,
            }
            mode_points.append(point)

            if log_path is not None:
                _append_jsonl(
                    log_path,
                    {
                        "event": "budget_aggregated",
                        "mode": mode,
                        **point,
                    },
                )
        curves_by_mode[mode] = mode_points

    if log_path is not None:
        _append_jsonl(
            log_path,
            {
                "event": "ablation_finished",
                "total_elapsed_sec": time.perf_counter() - experiment_started_at,
            },
        )

    return curves_by_mode

def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-driven multi-pattern dialogue loop.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--num-samples", type=int, default=10)
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
    parser.add_argument("--plot-ablation", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--ablation-log", type=str, default="")
    parser.add_argument("--budget-list", type=str, default="1,3,5")
    parser.add_argument("--modes", type=str, default="",
                        help="Comma-separated modes for --plot-ablation, e.g. 'direct_querying,user_preference_first'. Default: all 4.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting episode index (default 0). Use with --num-samples to select a contiguous range.")
    parser.add_argument("--sample-indices", type=str, default="",
                        help="Comma-separated explicit episode indices, e.g. '3,14,35,81,94'. Overrides --num-samples and --start-index.")
    parser.add_argument(
        "--selection-method",
        type=str,
        default="rule",
        choices=["rule", "entropy", "llm"],
        help="Pattern selection method: rule (deterministic), entropy (belief-driven), llm (original controller)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if args.sample_indices:
        sample_indices = [int(i.strip()) for i in args.sample_indices.split(",") if i.strip()]
    else:
        sample_indices = _select_sample_indices(num_samples=args.num_samples, start_index=args.start_index)
    episode = get_episode(data_path, sample_indices[0])
    budgets = _parse_budget_list(args.budget_list)

    if args.plot_ablation:
        output_path = args.output or f"plots/policy_ablation_{len(sample_indices)}ep.png"
        log_path = Path(args.ablation_log) if args.ablation_log else Path(output_path).with_suffix(".jsonl")
        if log_path.exists():
            log_path.unlink()
        ablation_modes = [m.strip() for m in args.modes.split(",") if m.strip()] if args.modes else None
        curves_by_mode = run_ablation_experiment(
            data_path=data_path,
            sample_indices=sample_indices,
            budgets=budgets,
            proposer_model=args.proposer_model,
            oracle_model=args.oracle_model,
            updater_model=args.updater_model,
            evaluation_model=args.evaluation_model,
            base_url=args.base_url,
            log_path=log_path,
            selection_method=args.selection_method,
            modes=ablation_modes,
        )
        episode_word = "episode" if len(sample_indices) == 1 else "episodes"
        saved_path = plot_ablation_comparison(
            curves_by_mode,
            output_path=output_path,
            title=f"Policy Ablation Across Budgets ({len(sample_indices)} {episode_word})",
            mode_labels=POLICY_LABELS,
        )
        print(json.dumps({"num_samples": len(sample_indices), "sample_indices": sample_indices, "curves_by_mode": curves_by_mode, "saved_plot": saved_path, "saved_log": str(log_path)}, indent=2, ensure_ascii=False))
        return

    if args.plot_curve:
        curve_points: List[Dict[str, Any]] = []
        for budget in budgets:
            seen_scores: List[float] = []
            unseen_scores: List[float] = []
            for index in sample_indices:
                sample_episode = get_episode(data_path, index)
                _, evaluation = run_policy_episode(
                    episode=sample_episode,
                    budget=budget,
                    mode=args.mode,
                    proposer_model=args.proposer_model,
                    oracle_model=args.oracle_model,
                    updater_model=args.updater_model,
                    evaluation_model=args.evaluation_model,
                    base_url=args.base_url,
                    verbose=False,
                    selection_method=args.selection_method,
                )
                seen_scores.append(float(evaluation["seen_accuracy"]))
                unseen_scores.append(float(evaluation["unseen_accuracy"]))

            curve_points.append(
                {
                    "budget": budget,
                    "seen_accuracy": sum(seen_scores) / len(seen_scores),
                    "unseen_accuracy": sum(unseen_scores) / len(unseen_scores),
                }
            )

        output_path = args.output or f"plots/{args.mode}_policy_accuracy_curve.png"
        episode_word = "sample" if len(sample_indices) == 1 else "samples"
        saved_path = plot_accuracy_curve(
            curve_points,
            output_path=output_path,
            title=f"{args.mode} Accuracy vs Budget ({len(sample_indices)} {episode_word})",
        )
        print(json.dumps({"num_samples": len(sample_indices), "sample_indices": sample_indices, "curve_points": curve_points, "saved_plot": saved_path}, indent=2, ensure_ascii=False))
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
            selection_method=args.selection_method,
        )

        print(f"=== Budget {budget} Final State ===")
        print(json.dumps(final_state, indent=2, ensure_ascii=False))
        print()

        print(f"=== Budget {budget} Evaluation ===")
        print(json.dumps(evaluation, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
