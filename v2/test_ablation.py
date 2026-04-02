"""
Ablation comparison: Raw LLM vs AO vs PE vs PI (cold-start) vs PI (seeded).

Runs all modes on the same sample set and generates a side-by-side accuracy plot
using plot_ablation_comparison from evaluation.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    from v2.data import DEFAULT_DATA_PATH, load_episodes, get_episode
    from v2.evaluation import plot_ablation_comparison
    from v2.test_question_pattern_loop import (
        run_question_pattern_experiment,
        _select_sample_indices,
        _parse_budget_list,
    )
    from v2.test_raw_llm import run_raw_llm_experiment
except ModuleNotFoundError:
    from data import DEFAULT_DATA_PATH, load_episodes, get_episode
    from evaluation import plot_ablation_comparison
    from test_question_pattern_loop import (
        run_question_pattern_experiment,
        _select_sample_indices,
        _parse_budget_list,
    )
    from test_raw_llm import run_raw_llm_experiment


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

MODE_LABELS = {
    "raw_llm": "Raw LLM",
    "action_oriented": "Action-Oriented",
    "preference_eliciting": "Preference-Eliciting",
    "preference_induction_cold": "Pref-Induction (cold)",
    "preference_induction_seeded": "Pref-Induction (seeded)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation comparison across all question patterns.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--budget-list", type=str, default="2,4,6")
    parser.add_argument("--model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    parser.add_argument(
        "--modes",
        type=str,
        default="raw_llm,action_oriented,preference_eliciting,preference_induction_cold,preference_induction_seeded",
        help="Comma-separated list of modes to run.",
    )
    parser.add_argument("--output", type=str, default="v2/plots/ablation_comparison.png")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    episodes = load_episodes(data_path)
    budgets = _parse_budget_list(args.budget_list)
    sample_indices = _select_sample_indices(
        num_samples=args.num_samples,
        total_episodes=len(episodes),
        sample_seed=args.sample_seed,
    )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"Running ablation: {modes}", flush=True)
    print(f"Samples: {sample_indices}", flush=True)
    print(f"Budgets: {budgets}", flush=True)

    curves_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    for mode in modes:
        print(f"\n{'='*60}", flush=True)
        print(f"Mode: {mode}", flush=True)
        print(f"{'='*60}", flush=True)

        if mode == "raw_llm":
            results = run_raw_llm_experiment(
                data_path=data_path,
                sample_indices=sample_indices,
                budgets=budgets,
                model=args.model,
                base_url=args.base_url,
                verbose=args.verbose,
            )
        elif mode in ("action_oriented", "preference_eliciting"):
            results = run_question_pattern_experiment(
                pattern=mode,  # type: ignore[arg-type]
                data_path=data_path,
                sample_indices=sample_indices,
                budgets=budgets,
                model=args.model,
                base_url=args.base_url,
                verbose=args.verbose,
                seed_induction=True,
            )
        elif mode == "preference_induction_cold":
            results = run_question_pattern_experiment(
                pattern="preference_induction",
                data_path=data_path,
                sample_indices=sample_indices,
                budgets=budgets,
                model=args.model,
                base_url=args.base_url,
                verbose=args.verbose,
                seed_induction=False,
            )
        elif mode == "preference_induction_seeded":
            results = run_question_pattern_experiment(
                pattern="preference_induction",
                data_path=data_path,
                sample_indices=sample_indices,
                budgets=budgets,
                model=args.model,
                base_url=args.base_url,
                verbose=args.verbose,
                seed_induction=True,
            )
        else:
            print(f"Unknown mode '{mode}', skipping.", flush=True)
            continue

        curves_by_mode[mode] = results["curve_points"]
        print(f"\n{mode} results:", flush=True)
        for pt in results["curve_points"]:
            print(
                f"  budget={pt['budget']}  seen={pt['seen_accuracy']:.4f}  unseen={pt['unseen_accuracy']:.4f}",
                flush=True,
            )

    saved_path = plot_ablation_comparison(
        curves_by_mode,
        output_path=args.output,
        title="Question Pattern Ablation",
        mode_labels=MODE_LABELS,
    )
    print(f"\nPlot saved: {saved_path}", flush=True)
    print(json.dumps({"curves_by_mode": curves_by_mode, "saved_plot": saved_path}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
