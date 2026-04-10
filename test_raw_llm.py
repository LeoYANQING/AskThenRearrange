"""
Raw LLM baseline for household rearrangement.

Two-step process per episode:
  1. Questioning loop: a free-form LLM generates one question per budget step;
     the same NaturalUserOracle answers each question.
  2. Placement: after budget is exhausted, a second LLM reads the full qa_history
     and outputs seen + unseen placements directly — no structured intermediate state.

When budget=0 ("no question" baseline), step 1 is skipped entirely.

Usage:
  # No question baseline (budget=0):
  LLM_BACKEND=openai LLM_MODEL=gpt-5-chat LLM_API_KEY=sk-xxx LLM_BASE_URL=https://api.gptsapi.net/v1 \
    python test_raw_llm.py --budget-list 0 --num-samples 102

  # Raw LLM with questions:
  LLM_BACKEND=openai LLM_MODEL=gpt-5-chat LLM_API_KEY=sk-xxx LLM_BASE_URL=https://api.gptsapi.net/v1 \
    python test_raw_llm.py --budget-list 0,1,3,5 --num-samples 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from data import DEFAULT_DATA_PATH, Episode, PlacementMap, get_episode, load_episodes
from evaluation import evaluate_episode_predictions, plot_accuracy_curve
from llm_factory import create_chat_model, DEFAULT_MODEL, DEFAULT_BASE_URL
from oracle import NaturalUserOracle


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class NextQuestion(BaseModel):
    question: str = Field(description="The next question to ask the user.")


class RawPlacementPlan(BaseModel):
    seen_placements: Dict[str, str] = Field(
        default_factory=dict,
        description="Placement for each seen object: {object_name: receptacle}.",
    )
    unseen_placements: Dict[str, str] = Field(
        default_factory=dict,
        description="Placement for each unseen object: {object_name: receptacle}.",
    )


# ---------------------------------------------------------------------------
# Raw LLM questioner
# ---------------------------------------------------------------------------

class RawLLMQuestioner:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> None:
        self._model = create_chat_model(model=model, base_url=base_url, temperature=0.0, timeout=120)
        self._structured = self._model.with_structured_output(NextQuestion)

    def next_question(
        self,
        *,
        room: str,
        receptacles: List[str],
        seen_objects: List[str],
        unseen_objects: List[str],
        qa_history: List[Dict[str, str]],
        budget_remaining: int,
    ) -> str:
        recent = [
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in qa_history[-6:]
            if item.get("question") and item.get("answer")
        ]
        system_prompt = """
You are an agent helping to rearrange a household room.
Your goal is to learn the user's placement preferences by asking questions.

- ask only about objects from the seen_objects or unseen_objects lists — never ask about receptacles
- do not repeat questions already answered
- use exact receptacle names from the provided list when relevant

Return exactly one question.
""".strip()

        user_prompt = f"""
Room: {room}

Receptacles: {receptacles}

Seen objects (visible in the room): {seen_objects}

Unseen objects (not yet encountered): {unseen_objects}

Questions asked so far:
{chr(10).join(recent) if recent else "(none yet)"}

Budget remaining: {budget_remaining} question(s)

Ask the next question about the user's placement preferences.
""".strip()

        result = self._structured.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return result.question.strip()


# ---------------------------------------------------------------------------
# Raw LLM placer
# ---------------------------------------------------------------------------

class RawLLMPlacer:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> None:
        self._model = create_chat_model(model=model, base_url=base_url, temperature=0.0, timeout=120)
        self._structured = self._model.with_structured_output(RawPlacementPlan)

    def place(
        self,
        *,
        room: str,
        receptacles: List[str],
        seen_objects: List[str],
        unseen_objects: List[str],
        qa_history: List[Dict[str, str]],
    ) -> Tuple[PlacementMap, PlacementMap]:
        qa_text = "\n".join(
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in qa_history
            if item.get("question") and item.get("answer")
        ) or "(no questions asked)"

        system_prompt = """
You are completing the final placement plan for a household rearrangement task.
Use the conversation history as your evidence to assign every object to a receptacle.

For seen objects — apply in order:
1. Direct mentions: if the conversation explicitly places an object, use that receptacle.
2. Category rules: if the conversation states a rule covering this object's type, apply it.
3. Frequency fallback: assign the receptacle most consistent with the conversation overall.

For unseen objects — apply in order:
1. Category rule match: find a rule from the conversation whose category fits this object; use its receptacle.
2. Analogy: find the most semantically similar seen object mentioned in the conversation; use the same receptacle.
3. Frequency fallback: assign the receptacle most consistent with the conversation overall.

Rules:
- use only exact object names from seen_objects and unseen_objects
- use only exact receptacle names from the provided receptacles list
- assign every object — leave no object unassigned
- prefer a specific stated rule over a generic guess
""".strip()

        user_prompt = f"""
Room: {room}

Receptacles: {receptacles}

Seen objects: {seen_objects}

Unseen objects: {unseen_objects}

Conversation history:
{qa_text}

Output a complete placement plan for ALL seen and unseen objects.
""".strip()

        result = self._structured.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        seen_plan = _normalize_placements(result.seen_placements, seen_objects, receptacles)
        unseen_plan = _normalize_placements(result.unseen_placements, unseen_objects, receptacles)
        return seen_plan, unseen_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_placements(
    placements: Dict[str, str],
    target_objects: List[str],
    receptacles: List[str],
) -> PlacementMap:
    allowed_objects = set(target_objects)
    allowed_receptacles = set(receptacles)
    return {
        obj: rec
        for obj, rec in placements.items()
        if obj in allowed_objects and rec in allowed_receptacles
    }


def _select_sample_indices(*, num_samples: int, total_episodes: int, sample_seed: int) -> List[int]:
    if num_samples >= total_episodes:
        return list(range(total_episodes))
    rng = random.Random(sample_seed)
    return sorted(rng.sample(list(range(total_episodes)), num_samples))


def _parse_index_range(value: str, total_episodes: int) -> List[int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"--index-range must be 'start:end', got '{value}'")
    start, end = int(parts[0]), int(parts[1])
    if not (0 <= start < end <= total_episodes):
        raise ValueError(f"--index-range {value} out of bounds for {total_episodes} episodes")
    return list(range(start, end))


def _parse_budget_list(value: str) -> List[int]:
    budgets = [int(b.strip()) for b in value.split(",") if b.strip()]
    if not budgets:
        raise ValueError("budget list must not be empty")
    for b in budgets:
        if b < 0:
            raise ValueError(f"budget values must be >= 0, got {b}")
    return budgets


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_raw_llm_episode(
    *,
    episode: Episode,
    budget: int,
    questioner: RawLLMQuestioner,
    oracle: NaturalUserOracle,
    placer: RawLLMPlacer,
    verbose: bool,
    eval_budgets: List[int] | None = None,
) -> Dict[str, Any] | Dict[int, Dict[str, Any]]:
    """Run a raw LLM episode.

    If eval_budgets is provided, run up to max(eval_budgets) steps and evaluate
    at each budget checkpoint. Returns {budget: metrics}.
    Otherwise runs to `budget` and returns metrics directly.
    """
    max_budget = max(eval_budgets) if eval_budgets else budget
    eval_set = set(eval_budgets) if eval_budgets else None
    qa_history: List[Dict[str, str]] = []
    results_by_budget: Dict[int, Dict[str, Any]] = {}

    # Evaluate at budget=0 if requested
    if eval_set is not None and 0 in eval_set:
        seen_plan, unseen_plan = placer.place(
            room=episode.room,
            receptacles=episode.receptacles,
            seen_objects=episode.seen_objects,
            unseen_objects=episode.unseen_objects,
            qa_history=qa_history,
        )
        metrics = evaluate_episode_predictions(
            episode, predicted_seen=seen_plan, predicted_unseen=unseen_plan,
        )
        results_by_budget[0] = {**metrics, "qa_history": list(qa_history)}

    for step in range(max_budget):
        budget_remaining = max_budget - step

        t0 = perf_counter()
        question = questioner.next_question(
            room=episode.room,
            receptacles=episode.receptacles,
            seen_objects=episode.seen_objects,
            unseen_objects=episode.unseen_objects,
            qa_history=qa_history,
            budget_remaining=budget_remaining,
        )
        t_question = perf_counter() - t0

        t1 = perf_counter()
        oracle_response = oracle.answer(
            question=question,
            room=episode.room,
            receptacles=episode.receptacles,
            seen_objects=episode.seen_objects,
            annotator_notes=episode.annotator_notes,
            gt_seen_placements=episode.seen_placements,
            qa_history=qa_history,
        )
        t_oracle = perf_counter() - t1

        qa_history.append({"question": question, "answer": oracle_response.answer})

        if verbose:
            print(f"  [step {step + 1}/{max_budget}] Q: {question}", flush=True)
            print(f"           A: {oracle_response.answer}", flush=True)
            print(f"           timings: question={t_question:.2f}s, oracle={t_oracle:.2f}s", flush=True)

        current_budget = len(qa_history)
        if eval_set is not None and current_budget in eval_set:
            seen_plan, unseen_plan = placer.place(
                room=episode.room,
                receptacles=episode.receptacles,
                seen_objects=episode.seen_objects,
                unseen_objects=episode.unseen_objects,
                qa_history=qa_history,
            )
            metrics = evaluate_episode_predictions(
                episode, predicted_seen=seen_plan, predicted_unseen=unseen_plan,
            )
            results_by_budget[current_budget] = {**metrics, "qa_history": list(qa_history)}

    if eval_set is not None:
        # Fill any remaining eval_budgets that weren't reached
        for b in eval_budgets:
            if b not in results_by_budget:
                seen_plan, unseen_plan = placer.place(
                    room=episode.room,
                    receptacles=episode.receptacles,
                    seen_objects=episode.seen_objects,
                    unseen_objects=episode.unseen_objects,
                    qa_history=qa_history,
                )
                metrics = evaluate_episode_predictions(
                    episode, predicted_seen=seen_plan, predicted_unseen=unseen_plan,
                )
                results_by_budget[b] = {**metrics, "qa_history": list(qa_history)}
        return results_by_budget

    # Single-budget mode
    seen_plan, unseen_plan = placer.place(
        room=episode.room,
        receptacles=episode.receptacles,
        seen_objects=episode.seen_objects,
        unseen_objects=episode.unseen_objects,
        qa_history=qa_history,
    )
    metrics = evaluate_episode_predictions(
        episode, predicted_seen=seen_plan, predicted_unseen=unseen_plan,
    )
    if verbose:
        print(f"  seen_accuracy={metrics['seen_accuracy']:.4f}  unseen_accuracy={metrics['unseen_accuracy']:.4f}", flush=True)

    return {**metrics, "qa_history": qa_history}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_raw_llm_experiment(
    *,
    data_path: Path,
    sample_indices: List[int],
    budgets: List[int],
    model: str,
    base_url: str,
    verbose: bool,
    output_jsonl: Path | None = None,
) -> Dict[str, Any]:
    questioner = RawLLMQuestioner(model=model, base_url=base_url)
    oracle = NaturalUserOracle(model=model, base_url=base_url, temperature=0.0)
    placer = RawLLMPlacer(model=model, base_url=base_url)

    total_episodes = len(sample_indices)
    experiment_start = perf_counter()

    if output_jsonl is not None:
        _append_jsonl(output_jsonl, {
            "event": "experiment_started",
            "sample_indices": sample_indices,
            "budgets": budgets,
        })

    # Per-budget accumulators
    budget_scores: Dict[int, Dict[str, List[float]]] = {
        b: {"seen": [], "unseen": []} for b in budgets
    }

    for ep_idx, index in enumerate(sample_indices):
        episode = get_episode(data_path, index)
        t_start = perf_counter()

        results_by_budget = run_raw_llm_episode(
            episode=episode,
            budget=max(budgets),
            questioner=questioner,
            oracle=oracle,
            placer=placer,
            verbose=verbose,
            eval_budgets=budgets,
        )
        elapsed = perf_counter() - t_start

        for b in budgets:
            ev = results_by_budget[b]
            budget_scores[b]["seen"].append(float(ev["seen_accuracy"]))
            budget_scores[b]["unseen"].append(float(ev["unseen_accuracy"]))

            if output_jsonl is not None:
                _append_jsonl(output_jsonl, {
                    "event": "episode_finished",
                    "mode": "no_question" if b == 0 else "raw_llm",
                    "budget": b,
                    "episode_index": index,
                    "episode_id": episode.episode_id,
                    "seen_accuracy": float(ev["seen_accuracy"]),
                    "unseen_accuracy": float(ev["unseen_accuracy"]),
                    "elapsed_sec": elapsed,
                })

        done = ep_idx + 1
        total_elapsed = perf_counter() - experiment_start
        eta = total_elapsed / done * (total_episodes - done)
        if done % 10 == 0 or done == 1:
            # Show result at max budget for progress
            max_ev = results_by_budget[max(budgets)]
            print(
                f"[{done}/{total_episodes}] ep{index}: "
                f"seen={max_ev['seen_accuracy']:.3f} unseen={max_ev['unseen_accuracy']:.3f} "
                f"({elapsed:.1f}s) | total {total_elapsed:.0f}s | ETA {eta:.0f}s ({eta/60:.0f}min)",
                flush=True,
            )

    # Aggregate
    curve_points: List[Dict[str, Any]] = []
    for budget in budgets:
        seen_scores = budget_scores[budget]["seen"]
        unseen_scores = budget_scores[budget]["unseen"]
        seen_mean = sum(seen_scores) / len(seen_scores)
        unseen_mean = sum(unseen_scores) / len(unseen_scores)
        n = len(seen_scores)
        if n > 1:
            seen_stderr = math.sqrt(sum((v - seen_mean) ** 2 for v in seen_scores) / (n - 1)) / math.sqrt(n)
            unseen_stderr = math.sqrt(sum((v - unseen_mean) ** 2 for v in unseen_scores) / (n - 1)) / math.sqrt(n)
        else:
            seen_stderr = 0.0
            unseen_stderr = 0.0

        point = {
            "budget": budget,
            "seen_accuracy": seen_mean,
            "unseen_accuracy": unseen_mean,
            "seen_stderr": seen_stderr,
            "unseen_stderr": unseen_stderr,
            "num_samples": n,
        }
        curve_points.append(point)

        if output_jsonl is not None:
            _append_jsonl(output_jsonl, {
                "event": "budget_aggregated",
                "mode": "no_question" if budget == 0 else "raw_llm",
                **point,
            })

    total_time = perf_counter() - experiment_start
    if output_jsonl is not None:
        _append_jsonl(output_jsonl, {
            "event": "experiment_finished",
            "total_elapsed_sec": total_time,
        })

    return {
        "pattern": "raw_llm",
        "sample_indices": sample_indices,
        "curve_points": curve_points,
        "total_elapsed_sec": total_time,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Raw LLM baseline for household rearrangement.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting episode index (default 0). Use with --num-samples.")
    parser.add_argument("--index-range", type=str, default="",
                        help="Explicit episode range 'start:end' (overrides --num-samples).")
    parser.add_argument("--sample-indices", type=str, default="",
                        help="Comma-separated explicit episode indices, e.g. '3,14,35'. Overrides other selection.")
    parser.add_argument("--budget-list", type=str, default="0")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--output-jsonl", type=str, default="",
                        help="Path to append per-episode results as JSONL.")
    parser.add_argument("--output-plot", type=str, default="",
                        help="Path to save accuracy curve plot.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    episodes = load_episodes(data_path)
    budgets = _parse_budget_list(args.budget_list)

    if args.sample_indices:
        sample_indices = [int(i.strip()) for i in args.sample_indices.split(",") if i.strip()]
    elif args.index_range:
        sample_indices = _parse_index_range(args.index_range, len(episodes))
    else:
        if args.start_index > 0:
            sample_indices = list(range(args.start_index, min(args.start_index + args.num_samples, len(episodes))))
        else:
            sample_indices = _select_sample_indices(
                num_samples=args.num_samples,
                total_episodes=len(episodes),
                sample_seed=args.sample_seed,
            )

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None

    results = run_raw_llm_experiment(
        data_path=data_path,
        sample_indices=sample_indices,
        budgets=budgets,
        model=args.model,
        base_url=args.base_url,
        verbose=args.verbose,
        output_jsonl=output_jsonl,
    )

    print(f"\n=== Summary ===")
    for pt in results["curve_points"]:
        label = "no_question" if pt["budget"] == 0 else f"raw_llm(b={pt['budget']})"
        print(f"  {label}: seen={pt['seen_accuracy']:.4f}±{pt['seen_stderr']:.4f}  unseen={pt['unseen_accuracy']:.4f}±{pt['unseen_stderr']:.4f}")
    print(f"  total time: {results['total_elapsed_sec']:.0f}s ({results['total_elapsed_sec']/60:.1f}min)")

    if args.output_plot and len(budgets) > 1:
        saved_path = plot_accuracy_curve(
            results["curve_points"],
            output_path=args.output_plot,
            title=f"Raw LLM Accuracy vs Budget ({len(sample_indices)} episodes)",
        )
        print(f"  plot saved to: {saved_path}")


if __name__ == "__main__":
    main()
