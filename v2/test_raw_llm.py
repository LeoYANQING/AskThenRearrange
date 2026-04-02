"""
Raw LLM baseline for household rearrangement.

Two-step process per episode:
  1. Questioning loop: a free-form LLM generates one question per budget step;
     the same NaturalUserOracle answers each question.
  2. Placement: after budget is exhausted, a second LLM reads the full qa_history
     and outputs seen + unseen placements directly — no structured intermediate state.

Used as a comparison baseline against pattern-based policies.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.data import DEFAULT_DATA_PATH, Episode, PlacementMap, get_episode, load_episodes
    from v2.evaluation import evaluate_episode_predictions, placement_accuracy, plot_accuracy_curve
    from v2.oracle import NaturalUserOracle, OracleResponse
except ModuleNotFoundError:
    from data import DEFAULT_DATA_PATH, Episode, PlacementMap, get_episode, load_episodes
    from evaluation import evaluate_episode_predictions, placement_accuracy, plot_accuracy_curve
    from oracle import NaturalUserOracle, OracleResponse


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


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
    def __init__(self, model: str = QUESTION_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
        self._model = ChatOllama(model=model, base_url=base_url, temperature=0.0, reasoning=False)
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
    def __init__(self, model: str = QUESTION_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
        self._model = ChatOllama(model=model, base_url=base_url, temperature=0.0, reasoning=False)
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
    """Parse 'start:end' into a list of indices [start, end)."""
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
) -> Dict[str, Any]:
    qa_history: List[Dict[str, str]] = []

    for step in range(budget):
        budget_remaining = budget - step

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
            print(f"  [step {step + 1}/{budget}] Q: {question}", flush=True)
            print(f"           A: {oracle_response.answer}", flush=True)
            print(f"           timings: question={t_question:.2f}s, oracle={t_oracle:.2f}s", flush=True)

    t2 = perf_counter()
    seen_plan, unseen_plan = placer.place(
        room=episode.room,
        receptacles=episode.receptacles,
        seen_objects=episode.seen_objects,
        unseen_objects=episode.unseen_objects,
        qa_history=qa_history,
    )
    t_place = perf_counter() - t2

    metrics = evaluate_episode_predictions(
        episode,
        predicted_seen=seen_plan,
        predicted_unseen=unseen_plan,
    )
    if verbose:
        print(f"  placement time: {t_place:.2f}s", flush=True)
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

    curve_points: List[Dict[str, Any]] = []
    for budget in budgets:
        seen_scores: List[float] = []
        unseen_scores: List[float] = []
        for pos, index in enumerate(sample_indices):
            episode = get_episode(data_path, index)
            print(
                f"=== Sample {pos + 1}/{len(sample_indices)} | raw_llm | budget={budget} | "
                f"index={index} | id={episode.episode_id} | room={episode.room} ===",
                flush=True,
            )
            t_start = perf_counter()
            result = run_raw_llm_episode(
                episode=episode,
                budget=budget,
                questioner=questioner,
                oracle=oracle,
                placer=placer,
                verbose=verbose,
            )
            elapsed = perf_counter() - t_start
            seen_scores.append(float(result["seen_accuracy"]))
            unseen_scores.append(float(result["unseen_accuracy"]))
            print(
                f"=== Result | id={episode.episode_id} | seen={result['seen_accuracy']:.4f} | "
                f"unseen={result['unseen_accuracy']:.4f} | total={elapsed:.1f}s ===",
                flush=True,
            )
            if output_jsonl is not None:
                _append_jsonl(output_jsonl, {
                    "event": "episode_finished",
                    "mode": "raw_llm",
                    "budget": budget,
                    "episode_index": index,
                    "episode_id": episode.episode_id,
                    "room": episode.room,
                    "seen_accuracy": float(result["seen_accuracy"]),
                    "unseen_accuracy": float(result["unseen_accuracy"]),
                    "elapsed_sec": elapsed,
                })

        curve_points.append(
            {
                "budget": budget,
                "seen_accuracy": sum(seen_scores) / len(seen_scores),
                "unseen_accuracy": sum(unseen_scores) / len(unseen_scores),
                "num_samples": len(sample_indices),
            }
        )

    return {"pattern": "raw_llm", "sample_indices": sample_indices, "curve_points": curve_points}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Raw LLM baseline for household rearrangement.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--index-range", type=str, default="",
                        help="Explicit episode range 'start:end' (overrides --num-samples/--sample-seed).")
    parser.add_argument("--budget-list", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    parser.add_argument("--output-jsonl", type=str, default="",
                        help="Path to append per-episode results as JSONL (for later merging).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    episodes = load_episodes(data_path)
    budgets = _parse_budget_list(args.budget_list)

    if args.index_range:
        sample_indices = _parse_index_range(args.index_range, len(episodes))
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

    output_path = "v2/plots/raw_llm_accuracy_curve.png"
    saved_path = plot_accuracy_curve(
        results["curve_points"],
        output_path=output_path,
        title="Raw LLM Accuracy vs Budget",
    )
    print(json.dumps({**results, "saved_plot": saved_path}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
