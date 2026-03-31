from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Literal, Tuple

try:
    from v2.agent_schema import AgentState
    from v2.data import DEFAULT_DATA_PATH, Episode, get_episode, load_episodes
    from v2.evaluation import FinalPlacementPlanner, evaluate_episode_state, plot_accuracy_curve
    from v2.oracle import NaturalUserOracle
    from v2.proposers import (
        ActionProposer,
        PreferenceElicitingProposer,
        PreferenceSummaryProposer,
        propose_preference_summary_intents,
    )
    from v2.state_init import build_initial_state
    from v2.state_update import StateUpdate
except ModuleNotFoundError:
    from agent_schema import AgentState
    from data import DEFAULT_DATA_PATH, Episode, get_episode, load_episodes
    from evaluation import FinalPlacementPlanner, evaluate_episode_state, plot_accuracy_curve
    from oracle import NaturalUserOracle
    from proposers import (
        ActionProposer,
        PreferenceElicitingProposer,
        PreferenceSummaryProposer,
        propose_preference_summary_intents,
    )
    from state_init import build_initial_state
    from state_update import StateUpdate


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
PatternName = Literal["action_oriented", "preference_eliciting", "preference_summary"]


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


def _sample_meta(*, pattern: PatternName, budget: int, index: int, episode: Episode) -> Dict[str, Any]:
    return {
        "pattern": pattern,
        "budget": budget,
        "sample_index": index,
        "sample_id": episode.episode_id,
        "room": episode.room,
    }


def _print_sample_started(
    *,
    pattern: PatternName,
    budget: int,
    episode: Episode,
    index: int,
    sample_position: int,
    num_samples: int,
) -> None:
    print(
        f"=== Sample {sample_position}/{num_samples} | pattern={pattern} | budget={budget} | "
        f"index={index} | id={episode.episode_id} | room={episode.room} ===",
        flush=True,
    )


def _print_stage(message: str) -> None:
    print(message, flush=True)


def _print_step_log(
    *,
    pattern: PatternName,
    budget: int,
    episode: Episode,
    index: int,
    step_log: Dict[str, Any],
    verbose: bool,
) -> None:
    step = step_log.get("step", "?")
    question_pattern = step_log.get("question_pattern", pattern)
    print(f"[Sample {episode.episode_id} | Question Step {step}] {question_pattern}", flush=True)
    if step_log.get("event") == "no_intent":
        print("  no intent available", flush=True)
        return

    intent = step_log.get("intent", {})
    oracle_response = step_log.get("oracle_response", {})
    state_snapshot = step_log.get("state", {})
    timings = step_log.get("timings", {})
    print(f"  question: {step_log.get('question', intent.get('question', ''))}", flush=True)
    print(f"  answer: {oracle_response.get('answer', '')}", flush=True)
    if timings:
        timing_text = ", ".join(
            f"{name}={value:.3f}s" for name, value in timings.items()
        )
        print(f"  timings: {timing_text}", flush=True)
    print(
        "  state:"
        f" budget_used={state_snapshot.get('budget_used')},"
        f" unresolved={len(state_snapshot.get('unresolved_objects', []))},"
        f" confirmed_actions={len(state_snapshot.get('confirmed_actions', []))},"
        f" confirmed_preferences={len(state_snapshot.get('confirmed_preferences', []))},"
        f" negative_preferences={len(state_snapshot.get('negative_preferences', []))},"
        f" negative_actions={len(state_snapshot.get('negative_actions', []))}",
        flush=True,
    )
    if not verbose:
        print(flush=True)
        return
    print("=== Current State ===", flush=True)
    print(json.dumps(state_snapshot, indent=2, ensure_ascii=False), flush=True)
    print(
        json.dumps(
            {
                "event": "step",
                **_sample_meta(
                    pattern=pattern,
                    budget=budget,
                    index=index,
                    episode=episode,
                ),
                **step_log,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(flush=True)


def _print_step_started(
    *,
    episode: Episode,
    step_idx: int,
    pattern: PatternName,
) -> None:
    _print_stage(
        f"[Sample {episode.episode_id} | Question Step {step_idx}] running {pattern}..."
    )


def _print_sample_finished(
    *,
    pattern: PatternName,
    budget: int,
    episode: Episode,
    index: int,
    evaluation: Dict[str, Any],
    verbose: bool,
    timings: Dict[str, float] | None = None,
) -> None:
    print(
        f"=== Sample Result | id={episode.episode_id} | seen={float(evaluation['seen_accuracy']):.4f} | "
        f"unseen={float(evaluation['unseen_accuracy']):.4f} ===",
        flush=True,
    )
    if timings:
        timing_text = ", ".join(
            f"{name}={value:.3f}s" for name, value in timings.items()
        )
        print(f"  timings: {timing_text}", flush=True)
    if verbose:
        print(
            json.dumps(
                {
                    "event": "sample_finished",
                    **_sample_meta(
                        pattern=pattern,
                        budget=budget,
                        index=index,
                        episode=episode,
                    ),
                    "seen_accuracy": float(evaluation["seen_accuracy"]),
                    "unseen_accuracy": float(evaluation["unseen_accuracy"]),
                    "timings": timings or {},
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )


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


def _select_sample_indices(*, num_samples: int, total_episodes: int, sample_seed: int) -> List[int]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if total_episodes <= 0:
        raise ValueError(f"total_episodes must be positive, got {total_episodes}")
    if num_samples >= total_episodes:
        return list(range(total_episodes))

    rng = random.Random(sample_seed)
    return sorted(rng.sample(list(range(total_episodes)), num_samples))


def _default_strategy_for_pattern(pattern: PatternName) -> str:
    if pattern == "action_oriented":
        return "direct"
    if pattern == "preference_eliciting":
        return "preference_first"
    return "parallel_exploration"


def _confirmed_action_map(state: AgentState) -> Dict[str, str]:
    return {
        item["object_name"]: item["receptacle"]
        for item in state["confirmed_actions"]
    }


def _seed_summary_state(state: AgentState, episode: Episode) -> AgentState:
    grouped: Dict[str, List[str]] = {}
    for obj, receptacle in episode.seen_placements.items():
        grouped.setdefault(receptacle, []).append(obj)

    for receptacle, objects in grouped.items():
        if len(objects) < 2:
            continue
        for obj in objects[:2]:
            state["confirmed_actions"].append({"object_name": obj, "receptacle": receptacle})

    confirmed_objects = set(_confirmed_action_map(state))
    state["unresolved_objects"] = [
        obj for obj in state["seen_objects"] if obj not in confirmed_objects
    ]
    return state


def _run_action_step(
    *,
    episode: Episode,
    state: AgentState,
    proposer: ActionProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
    step_idx: int,
) -> Tuple[AgentState, bool, Dict[str, Any] | None]:
    timings: Dict[str, float] = {}

    t0 = perf_counter()
    intent = proposer.propose(state=state)
    timings["propose"] = perf_counter() - t0
    if intent is None:
        return state, False, {
            "step": step_idx,
            "event": "no_intent",
            "question_pattern": "action_oriented",
            "timings": timings,
        }

    t1 = perf_counter()
    oracle_response = oracle.answer(
        question=intent.question,
        room=state["room"],
        receptacles=state["receptacles"],
        seen_objects=state["seen_objects"],
        annotator_notes=episode.annotator_notes,
        gt_seen_placements=episode.seen_placements,
        qa_history=state["qa_history"],
    )
    timings["oracle"] = perf_counter() - t1

    t2 = perf_counter()
    state = updater.update_state_from_action_answer(
        state=state,
        target=intent.object_name,
        answer=oracle_response.answer,
        question=intent.question,
        action_mode=intent.action_mode,
    )
    timings["update"] = perf_counter() - t2

    return state, True, {
        "step": step_idx,
        "question_pattern": "action_oriented",
        "intent": intent.model_dump(),
        "oracle_response": oracle_response.model_dump(),
        "state": _state_snapshot(state),
        "timings": timings,
    }


def _run_eliciting_step(
    *,
    episode: Episode,
    state: AgentState,
    proposer: PreferenceElicitingProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
    step_idx: int,
) -> Tuple[AgentState, bool, Dict[str, Any] | None]:
    timings: Dict[str, float] = {}

    t0 = perf_counter()
    intents = proposer.propose(
        state=state,
        max_intents=3,
    )
    timings["propose"] = perf_counter() - t0
    if not intents:
        return state, False, {
            "step": step_idx,
            "event": "no_intent",
            "question_pattern": "preference_eliciting",
            "timings": timings,
        }

    intent = intents[0]
    question = str(intent.get("question", ""))

    t2 = perf_counter()
    oracle_response = oracle.answer(
        question=question,
        room=state["room"],
        receptacles=state["receptacles"],
        seen_objects=state["seen_objects"],
        annotator_notes=episode.annotator_notes,
        gt_seen_placements=episode.seen_placements,
        qa_history=state["qa_history"],
    )
    timings["oracle"] = perf_counter() - t2

    t3 = perf_counter()
    state = updater.update_state_from_preference_eliciting_answer(
        state=state,
        hypothesis=str(intent.get("hypothesis", "")),
        covered_objects=list(intent.get("covered_objects", [])),
        answer=oracle_response.answer,
        question=question,
    )
    timings["update"] = perf_counter() - t3
    return state, True, {
        "step": step_idx,
        "question_pattern": "preference_eliciting",
        "intent": intent,
        "question": question,
        "oracle_response": oracle_response.model_dump(),
        "state": _state_snapshot(state),
        "timings": timings,
    }


def _run_summary_step(
    *,
    episode: Episode,
    state: AgentState,
    proposer: PreferenceSummaryProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
    step_idx: int,
) -> Tuple[AgentState, bool, Dict[str, Any] | None]:
    timings: Dict[str, float] = {}

    t0 = perf_counter()
    intents = propose_preference_summary_intents(
        state=state,
        proposer=proposer,
        max_intents=3,
    )
    timings["propose"] = perf_counter() - t0
    if not intents:
        return state, False, {
            "step": step_idx,
            "event": "no_intent",
            "question_pattern": "preference_summary",
            "timings": timings,
        }

    intent = intents[0]
    question = str(intent.get("question", ""))
    t1 = perf_counter()
    oracle_response = oracle.answer(
        question=question,
        room=state["room"],
        receptacles=state["receptacles"],
        seen_objects=state["seen_objects"],
        annotator_notes=episode.annotator_notes,
        gt_seen_placements=episode.seen_placements,
        qa_history=state["qa_history"],
    )
    timings["oracle"] = perf_counter() - t1

    t2 = perf_counter()
    state = updater.update_state_from_preference_summary_answer(
        state=state,
        hypothesis=str(intent.get("hypothesis", "")),
        covered_objects=list(intent.get("covered_objects", [])),
        answer=oracle_response.answer,
        question=question,
    )
    timings["update"] = perf_counter() - t2

    return state, True, {
        "step": step_idx,
        "question_pattern": "preference_summary",
        "intent": intent,
        "question": question,
        "oracle_response": oracle_response.model_dump(),
        "state": _state_snapshot(state),
        "timings": timings,
    }


def run_question_pattern_loop(
    *,
    pattern: PatternName,
    episode: Episode,
    budget: int,
    index: int,
    state: AgentState,
    action_proposer: ActionProposer,
    eliciting_proposer: PreferenceElicitingProposer,
    summary_proposer: PreferenceSummaryProposer,
    oracle: NaturalUserOracle,
    updater: StateUpdate,
    verbose: bool,
) -> Tuple[AgentState, List[Dict[str, Any]]]:
    step_idx = 0
    step_logs: List[Dict[str, Any]] = []
    while len(state["qa_history"]) < state["budget_total"]:
        step_idx += 1
        _print_step_started(
            episode=episode,
            step_idx=step_idx,
            pattern=pattern,
        )
        if pattern == "action_oriented":
            state, ok, step_log = _run_action_step(
                episode=episode,
                state=state,
                proposer=action_proposer,
                oracle=oracle,
                updater=updater,
                step_idx=step_idx,
            )
        elif pattern == "preference_eliciting":
            state, ok, step_log = _run_eliciting_step(
                episode=episode,
                state=state,
                proposer=eliciting_proposer,
                oracle=oracle,
                updater=updater,
                step_idx=step_idx,
            )
        else:
            state, ok, step_log = _run_summary_step(
                episode=episode,
                state=state,
                proposer=summary_proposer,
                oracle=oracle,
                updater=updater,
                step_idx=step_idx,
            )
        if step_log is not None:
            step_logs.append(step_log)
            _print_step_log(
                pattern=pattern,
                budget=budget,
                episode=episode,
                index=index,
                step_log=step_log,
                verbose=verbose,
            )
        if not ok:
            break
    return state, step_logs


def run_question_pattern_episode(
    *,
    pattern: PatternName,
    episode: Episode,
    index: int,
    budget: int,
    proposer_model: str,
    oracle_model: str,
    updater_model: str,
    evaluation_model: str,
    base_url: str,
    verbose: bool,
) -> tuple[AgentState, Dict[str, Any], List[Dict[str, Any]]]:
    sample_start = perf_counter()
    state = build_initial_state(
        episode=episode,
        strategy=_default_strategy_for_pattern(pattern),  # type: ignore[arg-type]
        budget_total=budget,
    )
    if pattern == "preference_summary":
        state = _seed_summary_state(state, episode)

    action_proposer = ActionProposer(
        model=proposer_model,
        base_url=base_url,
        temperature=0.0,
    )
    eliciting_proposer = PreferenceElicitingProposer(
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

    loop_start = perf_counter()
    final_state, step_logs = run_question_pattern_loop(
        pattern=pattern,
        episode=episode,
        budget=budget,
        index=index,
        state=state,
        action_proposer=action_proposer,
        eliciting_proposer=eliciting_proposer,
        summary_proposer=summary_proposer,
        oracle=oracle,
        updater=updater,
        verbose=verbose,
    )
    loop_seconds = perf_counter() - loop_start
    eval_start = perf_counter()
    evaluation = evaluate_episode_state(
        episode,
        final_state,
        planner=planner,
    )
    evaluation["timings"] = {
        "loop": loop_seconds,
        "evaluation": perf_counter() - eval_start,
        "total": perf_counter() - sample_start,
    }
    return final_state, evaluation, step_logs


def run_question_pattern_experiment(
    *,
    pattern: PatternName,
    data_path: Path,
    sample_indices: List[int],
    budgets: List[int],
    model: str,
    base_url: str,
    verbose: bool,
) -> Dict[str, Any]:
    curve_points: List[Dict[str, Any]] = []
    for budget in budgets:
        seen_scores: List[float] = []
        unseen_scores: List[float] = []
        for pos, index in enumerate(sample_indices):
            episode = get_episode(data_path, index)
            _print_sample_started(
                pattern=pattern,
                budget=budget,
                episode=episode,
                index=index,
                sample_position=pos + 1,
                num_samples=len(sample_indices),
            )
            _, evaluation, _step_logs = run_question_pattern_episode(
                pattern=pattern,
                episode=episode,
                index=index,
                budget=budget,
                proposer_model=model,
                oracle_model=model,
                updater_model=model,
                evaluation_model=model,
                base_url=base_url,
                verbose=verbose and len(sample_indices) == 1 and pos == 0,
            )
            seen_scores.append(float(evaluation["seen_accuracy"]))
            unseen_scores.append(float(evaluation["unseen_accuracy"]))
            _print_sample_finished(
                pattern=pattern,
                budget=budget,
                episode=episode,
                index=index,
                evaluation=evaluation,
                verbose=verbose,
                timings=evaluation.get("timings"),
            )

        curve_points.append(
            {
                "budget": budget,
                "seen_accuracy": sum(seen_scores) / len(seen_scores),
                "unseen_accuracy": sum(unseen_scores) / len(unseen_scores),
                "num_samples": len(sample_indices),
            }
        )
    return {
        "pattern": pattern,
        "sample_indices": sample_indices,
        "curve_points": curve_points,
    }


def main(
    default_pattern: PatternName | None = None,
    *,
    include_pattern_arg: bool = True,
) -> None:
    parser = argparse.ArgumentParser(
        description="Unified smoke test for question-pattern loops."
    )
    if include_pattern_arg:
        parser.add_argument(
            "--pattern",
            type=str,
            default=default_pattern or "action_oriented",
            choices=["action_oriented", "preference_eliciting", "preference_summary"],
        )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--budget-list", type=str, default="1,3,5")
    args = parser.parse_args()

    pattern = (
        args.pattern if include_pattern_arg else default_pattern or "action_oriented"
    )  # type: ignore[assignment]
    data_path = Path(args.data)
    episodes = load_episodes(data_path)
    budgets = _parse_budget_list(args.budget_list)
    sample_indices = _select_sample_indices(
        num_samples=args.num_samples,
        total_episodes=len(episodes),
        sample_seed=args.sample_seed,
    )
    results = run_question_pattern_experiment(
        pattern=pattern,
        data_path=data_path,
        sample_indices=sample_indices,
        budgets=budgets,
        model=args.model,
        base_url=args.base_url,
        verbose=args.verbose,
    )
    output_path = f"v2/plots/{pattern}_loop_accuracy_curve.png"
    saved_path = plot_accuracy_curve(
        results["curve_points"],
        output_path=output_path,
        title=f"{pattern.replace('_', '-')} Loop Accuracy vs Budget",
    )
    print(json.dumps({**results, "saved_plot": saved_path}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
