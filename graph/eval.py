from __future__ import annotations

from typing import Dict, List, TypedDict, Optional
from collections import defaultdict
import argparse
import json
from pathlib import Path


class EvalResult(TypedDict):
    seen_satisfaction: float
    unseen_satisfaction: float
    seen_correct: int
    seen_total: int
    unseen_correct: int
    unseen_total: int


def _placement_accuracy(
    objects: List[str],
    predicted: Dict[str, str],
    ground_truth: Dict[str, str],
    *,
    ignore_missing_predictions: bool = False,
) -> tuple[int, int, float]:
    """
    Computes object-level placement accuracy.

    Args:
        objects: The objects to evaluate (seen or unseen).
        predicted: object -> receptacle predicted by the agent.
        ground_truth: object -> receptacle ground truth.
        ignore_missing_predictions: If True, objects missing in `predicted` are excluded
            from the denominator. If False (default), missing predictions count as incorrect.

    Returns:
        (correct, total, accuracy)
    """
    if ignore_missing_predictions:
        eval_objects = [o for o in objects if o in predicted]
    else:
        eval_objects = list(objects)

    total = len(eval_objects)
    if total == 0:
        return 0, 0, 0.0

    correct = 0
    for obj in eval_objects:
        if obj in ground_truth and obj in predicted and predicted[obj] == ground_truth[obj]:
            correct += 1

    return correct, total, correct / total


def evaluate_episode(
    *,
    seen_objects: List[str],
    unseen_objects: List[str],
    predicted_seen: Dict[str, str],
    predicted_unseen: Dict[str, str],
    gt_seen: Dict[str, str],
    gt_unseen: Dict[str, str],
    ignore_missing_predictions: bool = False,
) -> EvalResult:
    """
    Minimal evaluator for one episode/sample.

    Use this outside LangGraph (Scheme A): the agent produces predictions,
    the evaluator compares them to GT and returns metrics.
    """
    seen_correct, seen_total, seen_acc = _placement_accuracy(
        seen_objects, predicted_seen, gt_seen, ignore_missing_predictions=ignore_missing_predictions
    )
    unseen_correct, unseen_total, unseen_acc = _placement_accuracy(
        unseen_objects, predicted_unseen, gt_unseen, ignore_missing_predictions=ignore_missing_predictions
    )

    return {
        "seen_satisfaction": seen_acc,
        "unseen_satisfaction": unseen_acc,
        "seen_correct": seen_correct,
        "seen_total": seen_total,
        "unseen_correct": unseen_correct,
        "unseen_total": unseen_total,
    }


def _normalize_placements(placements) -> Dict[str, str]:
    if placements is None:
        return {}
    if isinstance(placements, dict):
        return dict(placements)
    if isinstance(placements, list):
        out: Dict[str, str] = {}
        for item in placements:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = str(item[1])
        return out
    return {}


def _get_gt_seen(ep: dict) -> Dict[str, str]:
    if "gt_seen_placements" in ep:
        return _normalize_placements(ep.get("gt_seen_placements"))
    if "seen_placements" in ep:
        return _normalize_placements(ep.get("seen_placements"))
    if "seen_placement" in ep:
        return _normalize_placements(ep.get("seen_placement"))
    return {}


def _get_gt_unseen(ep: dict) -> Dict[str, str]:
    if "gt_unseen_placements" in ep:
        return _normalize_placements(ep.get("gt_unseen_placements"))
    if "unseen_placements" in ep:
        return _normalize_placements(ep.get("unseen_placements"))
    if "unseen_placement" in ep:
        return _normalize_placements(ep.get("unseen_placement"))
    return {}


def evaluate_dataset(
    episodes: List[dict],
    *,
    ignore_missing_predictions: bool = False,
) -> Dict[str, float]:
    """
    Aggregate metrics over a dataset.
    Each episode dict is expected to include:
      - seen_objects, unseen_objects
      - predicted_placements_seen, predicted_placements_unseen
      - gt_seen_placements, gt_unseen_placements
    """
    total_seen_correct = total_seen = 0
    total_unseen_correct = total_unseen = 0

    for ep in episodes:
        res = evaluate_episode(
            seen_objects=ep["seen_objects"],
            unseen_objects=ep.get("unseen_objects", []),
            predicted_seen=ep["predicted_placements_seen"],
            predicted_unseen=ep.get("predicted_placements_unseen", {}),
            gt_seen=_get_gt_seen(ep),
            gt_unseen=_get_gt_unseen(ep),
            ignore_missing_predictions=ignore_missing_predictions,
        )
        total_seen_correct += res["seen_correct"]
        total_seen += res["seen_total"]
        total_unseen_correct += res["unseen_correct"]
        total_unseen += res["unseen_total"]

    seen_satisfaction = (total_seen_correct / total_seen) if total_seen > 0 else 0.0
    unseen_satisfaction = (total_unseen_correct / total_unseen) if total_unseen > 0 else 0.0

    return {
        "seen_satisfaction": seen_satisfaction,
        "unseen_satisfaction": unseen_satisfaction,
        "seen_correct": float(total_seen_correct),
        "seen_total": float(total_seen),
        "unseen_correct": float(total_unseen_correct),
        "unseen_total": float(total_unseen),
    }


def _get_question_count(ep: dict, *, question_count_field: Optional[str] = None) -> Optional[int]:
    if question_count_field and question_count_field in ep:
        try:
            return int(ep[question_count_field])
        except (TypeError, ValueError):
            return None

    if "qa_history" in ep and isinstance(ep["qa_history"], list):
        return len(ep["qa_history"])

    if "budget_used" in ep:
        try:
            return int(ep["budget_used"])
        except (TypeError, ValueError):
            return None

    if "budget_total" in ep:
        try:
            return int(ep["budget_total"])
        except (TypeError, ValueError):
            return None

    return None


def plot_strategy_tradeoff(
    episodes: List[dict],
    *,
    output_path: str = "strategy_tradeoff.png",
    strategy_field: str = "strategy",
    question_count_field: Optional[str] = None,
    ignore_missing_predictions: bool = False,
) -> None:
    """
    Plot question count vs accuracy for different strategies.

    Each episode dict is expected to include:
      - seen_objects, unseen_objects
      - predicted_placements_seen, predicted_placements_unseen
      - gt_seen_placements, gt_unseen_placements
      - a strategy label (default field: "strategy")
      - qa_history or budget_used to infer question count
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {
            "seen_correct": 0,
            "seen_total": 0,
            "unseen_correct": 0,
            "unseen_total": 0,
        }
    )

    for ep in episodes:
        strategy = str(ep.get(strategy_field, "unknown"))
        q_count = _get_question_count(ep, question_count_field=question_count_field)
        if q_count is None:
            continue

        res = evaluate_episode(
            seen_objects=ep["seen_objects"],
            unseen_objects=ep.get("unseen_objects", []),
            predicted_seen=ep["predicted_placements_seen"],
            predicted_unseen=ep.get("predicted_placements_unseen", {}),
            gt_seen=_get_gt_seen(ep),
            gt_unseen=_get_gt_unseen(ep),
            ignore_missing_predictions=ignore_missing_predictions,
        )

        key = (strategy, q_count)
        grouped[key]["seen_correct"] += res["seen_correct"]
        grouped[key]["seen_total"] += res["seen_total"]
        grouped[key]["unseen_correct"] += res["unseen_correct"]
        grouped[key]["unseen_total"] += res["unseen_total"]

    strategies = sorted({s for s, _ in grouped.keys()})
    if not strategies:
        raise ValueError("No valid episodes found to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for strategy in strategies:
        counts = sorted({q for s, q in grouped.keys() if s == strategy})
        seen_accs = []
        unseen_accs = []
        for q in counts:
            data = grouped[(strategy, q)]
            seen_acc = (data["seen_correct"] / data["seen_total"]) if data["seen_total"] > 0 else 0.0
            unseen_acc = (data["unseen_correct"] / data["unseen_total"]) if data["unseen_total"] > 0 else 0.0
            seen_accs.append(seen_acc)
            unseen_accs.append(unseen_acc)

        axes[0].plot(counts, seen_accs, marker="o", label=strategy)
        axes[1].plot(counts, unseen_accs, marker="o", label=strategy)

    axes[0].set_title("Seen Accuracy")
    axes[1].set_title("Unseen Accuracy")
    for ax in axes:
        ax.set_xlabel("Question Count")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def _load_episodes_from_path(path: Path) -> List[dict]:
    if path.is_dir():
        episodes: List[dict] = []
        for item in sorted(path.glob("*.json")):
            episodes.extend(_load_episodes_from_path(item))
        return episodes

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON format in {path}")


def load_episodes(inputs: List[str]) -> List[dict]:
    episodes: List[dict] = []
    for item in inputs:
        episodes.extend(_load_episodes_from_path(Path(item)))
    return episodes
