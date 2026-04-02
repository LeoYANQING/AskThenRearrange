"""
Combine raw_llm baseline JSONL(s) + policy experiment JSONL(s) into a single
ablation comparison plot.

Both JSONL formats are accepted — records are filtered to event == "episode_finished".

Usage:
    python -m v2.compare_results \
        --files v2/logs/shard_0.jsonl v2/logs/shard_1.jsonl \
                v2/logs/shard_2.jsonl v2/logs/shard_3.jsonl \
                v2/logs/policy_ablation_10ep.jsonl \
        --modes raw_llm user_preference_first parallel_exploration \
        --output-plot v2/plots/comparison.png
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from v2.evaluation import plot_ablation_comparison
except ModuleNotFoundError:
    from evaluation import plot_ablation_comparison


MODE_LABELS = {
    "raw_llm": "Raw LLM (baseline)",
    "direct_querying": "Direct Querying",
    "user_preference_first": "User-Preference-First",
    "parallel_exploration": "Parallel Exploration",
    "hybrid_all": "Hybrid-All",
}


def load_episode_records(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load all JSONL files and keep only episode_finished records."""
    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("event") == "episode_finished":
                    records.append(record)
    return records


def aggregate_by_mode(
    records: List[Dict[str, Any]],
    modes: List[str] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by mode, aggregate per-budget accuracy, return curves_by_mode."""
    by_mode_budget: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        mode = r.get("mode", "unknown")
        if modes and mode not in modes:
            continue
        budget = int(r["budget"])
        by_mode_budget[mode][budget].append(r)

    curves_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for mode, budget_map in by_mode_budget.items():
        points: List[Dict[str, Any]] = []
        for budget in sorted(budget_map):
            entries = budget_map[budget]
            seen = [e["seen_accuracy"] for e in entries]
            unseen = [e["unseen_accuracy"] for e in entries]
            n = len(entries)
            seen_mean = sum(seen) / n
            unseen_mean = sum(unseen) / n
            if n > 1:
                seen_stderr = math.sqrt(
                    sum((x - seen_mean) ** 2 for x in seen) / (n - 1)
                ) / math.sqrt(n)
                unseen_stderr = math.sqrt(
                    sum((x - unseen_mean) ** 2 for x in unseen) / (n - 1)
                ) / math.sqrt(n)
            else:
                seen_stderr = unseen_stderr = 0.0
            points.append({
                "budget": budget,
                "seen_accuracy": seen_mean,
                "unseen_accuracy": unseen_mean,
                "seen_stderr": seen_stderr,
                "unseen_stderr": unseen_stderr,
                "num_episodes": n,
            })
        curves_by_mode[mode] = points

    return curves_by_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw_llm + policy results.")
    parser.add_argument("--files", nargs="+", required=True,
                        help="JSONL files to combine (raw_llm shards and/or policy logs).")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Which modes to include (default: all found).")
    parser.add_argument("--output-plot", type=str,
                        default="v2/plots/comparison.png")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    paths = [Path(p) for p in args.files]
    records = load_episode_records(paths)
    print(f"Loaded {len(records)} episode_finished records from {len(paths)} file(s).", flush=True)

    curves_by_mode = aggregate_by_mode(records, modes=args.modes)

    for mode, points in curves_by_mode.items():
        print(f"\n[{MODE_LABELS.get(mode, mode)}]")
        for pt in points:
            print(
                f"  budget={pt['budget']:2d}  "
                f"seen={pt['seen_accuracy']:.4f}±{pt['seen_stderr']:.4f}  "
                f"unseen={pt['unseen_accuracy']:.4f}±{pt['unseen_stderr']:.4f}  "
                f"n={pt['num_episodes']}"
            )

    n_episodes = max(
        (pt["num_episodes"] for pts in curves_by_mode.values() for pt in pts),
        default=0,
    )
    title = args.title or f"Method Comparison (n={n_episodes} episodes per budget)"
    saved = plot_ablation_comparison(
        curves_by_mode,
        output_path=args.output_plot,
        title=title,
        mode_labels=MODE_LABELS,
    )
    print(f"\nPlot saved: {saved}")
    print(json.dumps({"curves_by_mode": curves_by_mode, "saved_plot": saved}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
