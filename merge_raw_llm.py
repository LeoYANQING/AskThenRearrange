"""
Merge per-episode JSONL shards from parallel raw_llm workers into a single
accuracy curve and plot.

Usage:
    python -m merge_raw_llm \
        --shards logs/shard_0.jsonl logs/shard_1.jsonl ... \
        --output-plot plots/raw_llm_accuracy_curve.png
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from evaluation import plot_accuracy_curve


def load_shards(shard_paths: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in shard_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_budget: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_budget[r["budget"]].append(r)

    curve_points: List[Dict[str, Any]] = []
    for budget in sorted(by_budget):
        entries = by_budget[budget]
        seen_scores = [e["seen_accuracy"] for e in entries]
        unseen_scores = [e["unseen_accuracy"] for e in entries]
        n = len(entries)
        seen_mean = sum(seen_scores) / n
        unseen_mean = sum(unseen_scores) / n
        if n > 1:
            seen_stderr = math.sqrt(
                sum((x - seen_mean) ** 2 for x in seen_scores) / (n - 1)
            ) / math.sqrt(n)
            unseen_stderr = math.sqrt(
                sum((x - unseen_mean) ** 2 for x in unseen_scores) / (n - 1)
            ) / math.sqrt(n)
        else:
            seen_stderr = unseen_stderr = 0.0
        curve_points.append({
            "budget": budget,
            "seen_accuracy": seen_mean,
            "unseen_accuracy": unseen_mean,
            "seen_stderr": seen_stderr,
            "unseen_stderr": unseen_stderr,
            "num_episodes": n,
        })
    return curve_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge raw_llm JSONL shards.")
    parser.add_argument("--shards", nargs="+", required=True,
                        help="Paths to per-shard JSONL files.")
    parser.add_argument("--output-plot", type=str,
                        default="plots/raw_llm_accuracy_curve.png")
    args = parser.parse_args()

    shard_paths = [Path(p) for p in args.shards]
    records = load_shards(shard_paths)
    print(f"Loaded {len(records)} episode records from {len(shard_paths)} shards.", flush=True)

    curve_points = aggregate(records)
    for pt in curve_points:
        print(
            f"  budget={pt['budget']:2d}  "
            f"seen={pt['seen_accuracy']:.4f}±{pt['seen_stderr']:.4f}  "
            f"unseen={pt['unseen_accuracy']:.4f}±{pt['unseen_stderr']:.4f}  "
            f"n={pt['num_episodes']}",
            flush=True,
        )

    saved = plot_accuracy_curve(
        curve_points,
        output_path=args.output_plot,
        title=f"Raw LLM Accuracy vs Budget (n={curve_points[0]['num_episodes'] if curve_points else 0} per budget)",
    )
    print(json.dumps({"curve_points": curve_points, "saved_plot": saved}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
