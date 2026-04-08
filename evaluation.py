from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from llm_factory import create_chat_model, DEFAULT_MODEL, DEFAULT_BASE_URL
from pydantic import BaseModel, Field

from agent_schema import AgentState
from data import Episode, PlacementMap


# Model config now in llm_factory.py
# Base URL config now in llm_factory.py


class FinalPlacementPlan(BaseModel):
    placements: Dict[str, str] = Field(default_factory=dict)


class FinalPlacementPlanner:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model: Any = create_chat_model(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
            timeout=120,
        )
        self.structured_model = self.model.with_structured_output(FinalPlacementPlan)

    def plan_placements(
        self,
        *,
        state: AgentState,
        target_objects: List[str],
        scope: str,
    ) -> PlacementMap:
        if not target_objects:
            return {}

        if scope == "seen":
            system_prompt = """
You complete the final seen-object placement plan for a household rearrangement agent.

Your job: assign one receptacle to each remaining unresolved seen object.

CRITICAL — evidence always overrides your general knowledge:
- This user's household has specific rules captured in confirmed_actions and confirmed_preferences.
- Do NOT use general common sense (e.g. "clothes go in wardrobes", "reading items go on shelves") when confirmed evidence contradicts it.
- The confirmed evidence is ground truth for THIS household.

How to use the evidence:
1. confirmed_preferences (highest priority for category members): if a target object fits a preference's category, assign it to that preference's receptacle — even if common sense would suggest elsewhere.
2. confirmed_actions analogy (second priority): find the most semantically similar confirmed object by function or use context, not just by name keywords. Apply its receptacle.
3. Only fall back to general reasoning if no confirmed evidence applies at all.

Rules:
- return only exact target object names as keys
- return only exact receptacle names from the provided receptacles list
- do not choose any object -> receptacle pair listed in negative_actions
- make a complete plan for all target_objects
""".strip()
        else:
            system_prompt = """
You complete the final unseen-object placement plan for a household rearrangement agent.

Unseen objects belong in the same household and must follow the same rules the user expressed for seen objects.

CRITICAL — evidence always overrides your general knowledge:
- Do NOT use general common sense (e.g. "clothes go in wardrobes") when confirmed evidence says otherwise.
- This household's rules are captured in confirmed_actions and confirmed_preferences — trust them absolutely.

Inference strategy — apply strictly in order for each unseen object:

Step 1 — confirmed_preferences match (HIGHEST PRIORITY):
  Read every confirmed_preference. Each has a "hypothesis" (category description) and a "receptacle".
  Match by shared attributes: material, size, power source, or usage context.

  CRITICAL — when multiple preferences could match an object:
  - Compare the DISTINGUISHING words in each preference hypothesis.
  - Example: "large flat items for viewing/reading" vs "small books and puzzles for play/relaxation"
    → a "sudoku booklet" is SMALL and for PLAY → matches "small books for play" → center table, NOT display shelf
    → a "hardcover art book" is LARGE and for VIEWING → matches "large items for viewing" → display shelf
  - Focus on the adjectives (large/small, soft/hard, portable/plug-in) to break ties.
  - Focus on the PURPOSE (for reading vs for play, for storage vs for display, for comfort vs for organization).
  - If a preference says "miscellaneous items, loose accessories" — this is a CATCH-ALL; objects that don't clearly fit any other preference go here.

Step 2 — confirmed_actions analogy (use attributes and function, not name keywords):
  Find the most similar seen object in confirmed_actions by material, size, power source, or usage context.
  Use the same receptacle.
  Example: unseen "bamboo spatula" shares material (wood) and function (prep tool) with seen "wooden cutting board" → same receptacle.

Step 3 — receptacle frequency fallback:
  Only if neither Step 1 nor Step 2 applies, assign the most frequent receptacle for that object type.

Rules:
- return only exact target object names as keys
- return only exact receptacle names from the provided receptacles list
- do not choose any object -> receptacle pair listed in negative_actions
- make a complete plan for all target_objects
""".strip()

        if scope == "seen":
            user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Target objects (assign each to one receptacle):
{target_objects}

Confirmed actions (already decided placements):
{state["confirmed_actions"]}

Confirmed preferences (learned rules):
{state["confirmed_preferences"]}

Negative actions (forbidden pairs):
{state["negative_actions"]}
""".strip()
        else:
            user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects (for context — their organization reflects the user's preferences):
{state["seen_objects"]}

Confirmed actions for seen objects (use as analogy evidence):
{state["confirmed_actions"]}

Confirmed preferences (category-level rules — primary inference source):
{state["confirmed_preferences"]}

Target unseen objects (assign each to one receptacle by generalizing the rules above):
{target_objects}
""".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Retry on JSON parse failures (common with some models)
        last_exc = None
        for attempt in range(3):
            try:
                result = self.structured_model.invoke(messages)
                return _normalize_planned_placements(
                    result.placements,
                    target_objects=target_objects,
                    receptacles=state["receptacles"],
                    negative_actions=state["negative_actions"],
                )
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    continue
        # Final fallback: try unstructured model and parse Python dict
        try:
            import ast
            raw = self.model.invoke(messages)
            text = raw.content if hasattr(raw, "content") else str(raw)
            # Try to parse as Python dict (model sometimes returns single-quoted JSON)
            placements = ast.literal_eval(text.strip())
            if isinstance(placements, dict):
                return _normalize_planned_placements(
                    placements,
                    target_objects=target_objects,
                    receptacles=state["receptacles"],
                    negative_actions=state["negative_actions"],
                )
        except Exception:
            pass
        raise last_exc  # type: ignore[misc]


def _normalize_planned_placements(
    placements: Dict[str, str],
    *,
    target_objects: List[str],
    receptacles: List[str],
    negative_actions: List[dict],
) -> PlacementMap:
    target_set = set(target_objects)
    allowed_receptacles = set(receptacles)
    negative_pairs = {(item["object_name"], item["receptacle"]) for item in negative_actions}
    normalized: PlacementMap = {}
    for obj, receptacle in placements.items():
        if obj not in target_set:
            continue
        if receptacle not in allowed_receptacles:
            continue
        if (obj, receptacle) in negative_pairs:
            continue
        normalized[obj] = receptacle
    return normalized


def placement_accuracy(
    predicted: PlacementMap,
    gold: PlacementMap,
    objects: List[str],
) -> float:
    if not objects:
        return 1.0
    correct = 0
    for obj in objects:
        if predicted.get(obj) == gold.get(obj):
            correct += 1
    return correct / len(objects)


def finalize_seen_placements(
    state: AgentState,
    *,
    planner: FinalPlacementPlanner,
) -> PlacementMap:
    confirmed = {item["object_name"]: item["receptacle"] for item in state["confirmed_actions"]}
    remaining = [obj for obj in state["seen_objects"] if obj not in confirmed]
    planned = planner.plan_placements(state=state, target_objects=remaining, scope="seen")
    return {**confirmed, **planned}


def finalize_unseen_placements(
    state: AgentState,
    *,
    planner: FinalPlacementPlanner,
) -> PlacementMap:
    return planner.plan_placements(
        state=state,
        target_objects=state["unseen_objects"],
        scope="unseen",
    )


def evaluate_episode_predictions(
    episode: Episode,
    *,
    predicted_seen: PlacementMap,
    predicted_unseen: PlacementMap,
) -> Dict[str, Any]:
    return {
        "seen_accuracy": placement_accuracy(
            predicted_seen,
            episode.seen_placements,
            episode.seen_objects,
        ),
        "unseen_accuracy": placement_accuracy(
            predicted_unseen,
            episode.unseen_placements,
            episode.unseen_objects,
        ),
        "num_seen_objects": len(episode.seen_objects),
        "num_unseen_objects": len(episode.unseen_objects),
        "num_predicted_seen": len(predicted_seen),
        "num_predicted_unseen": len(predicted_unseen),
    }


def evaluate_episode_state(
    episode: Episode,
    state: AgentState,
    *,
    planner: FinalPlacementPlanner,
) -> Dict[str, Any]:
    finalized_seen = finalize_seen_placements(
        state,
        planner=planner,
    )
    finalized_unseen = finalize_unseen_placements(
        state,
        planner=planner,
    )
    metrics = evaluate_episode_predictions(
        episode,
        predicted_seen=finalized_seen,
        predicted_unseen=finalized_unseen,
    )
    return {
        **metrics,
        "finalized_placements_seen": finalized_seen,
        "finalized_placements_unseen": finalized_unseen,
    }



def plot_ablation_comparison(
    curves_by_mode: Dict[str, List[Dict[str, Any]]],
    *,
    output_path: str | Path,
    title: str,
    mode_labels: Dict[str, str] | None = None,
) -> str:
    mode_labels = mode_labels or {}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    palette = {
        "direct_querying": "#2f2f2f",
        "user_preference_first": "#1f77b4",
        "parallel_exploration": "#d55e00",
        "hybrid_all": "#2a9d8f",
        # pattern ablation modes
        "raw_llm": "#aaaaaa",
        "action_oriented": "#2f2f2f",
        "preference_eliciting": "#1f77b4",
        "preference_induction_cold": "#d55e00",
        "preference_induction_seeded": "#2a9d8f",
    }
    markers = {
        "direct_querying": "o",
        "user_preference_first": "s",
        "parallel_exploration": "^",
        "hybrid_all": "D",
        "raw_llm": "x",
        "action_oriented": "o",
        "preference_eliciting": "s",
        "preference_induction_cold": "^",
        "preference_induction_seeded": "D",
    }
    linestyles = {
        "direct_querying": "-",
        "user_preference_first": "-",
        "parallel_exploration": "-",
        "hybrid_all": "--",
        "raw_llm": ":",
        "action_oriented": "-",
        "preference_eliciting": "-",
        "preference_induction_cold": "--",
        "preference_induction_seeded": "-",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True, sharey=True)
    panel_specs = [
        ("seen_accuracy", "Seen Accuracy"),
        ("unseen_accuracy", "Unseen Accuracy"),
    ]
    all_budgets = sorted({int(point["budget"]) for points in curves_by_mode.values() for point in points})

    for ax, (metric_key, panel_title) in zip(axes, panel_specs):
        for mode, points in curves_by_mode.items():
            sorted_points = sorted(points, key=lambda item: int(item["budget"]))
            budgets = [int(point["budget"]) for point in sorted_points]
            values = [float(point[metric_key]) for point in sorted_points]
            stderr_key = metric_key.replace("accuracy", "stderr")
            stderr_values = [float(point.get(stderr_key, 0.0)) for point in sorted_points]
            color = palette.get(mode, "#444444")
            ax.plot(
                budgets,
                values,
                color=color,
                linestyle=linestyles.get(mode, "-"),
                linewidth=1.8,
                marker=markers.get(mode, "o"),
                markersize=5.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=mode_labels.get(mode, mode),
                zorder=3,
            )
            if any(value > 0.0 for value in stderr_values):
                lower = [max(0.0, mean - err) for mean, err in zip(values, stderr_values)]
                upper = [min(1.0, mean + err) for mean, err in zip(values, stderr_values)]
                ax.fill_between(budgets, lower, upper, color=color, alpha=0.12, linewidth=0.0, zorder=2)

        ax.set_title(panel_title, fontsize=12, pad=10)
        ax.set_xlabel("Budget", fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(FixedLocator(all_budgets))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        if all_budgets:
            ax.set_xlim(min(all_budgets), max(all_budgets))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        fontsize=10,
    )
    fig.suptitle(title, fontsize=13, y=1.08)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)

def plot_accuracy_curve(
    curve_points: List[Dict[str, Any]],
    *,
    output_path: str | Path,
    title: str,
) -> str:
    budgets = [int(point["budget"]) for point in curve_points]
    seen_acc = [float(point["seen_accuracy"]) for point in curve_points]
    unseen_acc = [float(point["unseen_accuracy"]) for point in curve_points]
    budget_ticks = list(dict.fromkeys(budgets))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.plot(
        budgets,
        seen_acc,
        color="black",
        linewidth=1.6,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Seen",
        zorder=3,
    )
    ax.plot(
        budgets,
        unseen_acc,
        color="#7a7a7a",
        linewidth=1.6,
        marker="s",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Unseen",
        zorder=3,
    )
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=12, pad=10)
    ax.xaxis.set_major_locator(FixedLocator(budget_ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_xlim(min(budget_ticks), max(budget_ticks))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)
