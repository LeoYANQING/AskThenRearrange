from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.agent_schema import AgentState
    from v2.data import Episode, PlacementMap
except ModuleNotFoundError:
    from agent_schema import AgentState
    from data import Episode, PlacementMap


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class FinalPlacementPlan(BaseModel):
    placements: Dict[str, str] = Field(default_factory=dict)


class FinalPlacementPlanner:
    def __init__(
        self,
        model: str = QUESTION_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model: Any = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self.structured_model = self.model.with_structured_output(FinalPlacementPlan)

    def plan_placements(
        self,
        *,
        state: AgentState,
        target_objects: List[str],
        scope: str,
        respect_exclusions: bool,
    ) -> PlacementMap:
        if not target_objects:
            return {}

        excluded_receptacles = state["excluded_receptacles"] if respect_exclusions else {}
        scope_instruction = (
            "assign one receptacle to each remaining unresolved seen object"
            if scope == "seen"
            else "assign one receptacle to each unseen object"
        )
        extra_instruction = (
            "use the current state as planning context"
            if scope == "seen"
            else "generalize conservatively from the current state"
        )
        exclusion_block = (
            f"\nExcluded receptacles:\n{state['excluded_receptacles']}"
            if respect_exclusions
            else ""
        )
        exclusion_rule = (
            "\n- respect excluded_receptacles for each object"
            if respect_exclusions
            else ""
        )

        system_prompt = f"""
You complete the final {scope}-object placement plan for a household rearrangement agent.

Your job:
- {scope_instruction}
- {extra_instruction}

You may use:
- room
- receptacles
- seen_objects
- target_objects
- confirmed_actions
- excluded_receptacles
- confirmed_preferences

Rules:
- return only exact target object names as keys
- return only exact receptacle names from the provided receptacles
- be consistent with confirmed_actions and confirmed_preferences
- make a complete plan for all target_objects{exclusion_rule}
- be consistent with confirmed_actions and confirmed_preferences
""".strip()

        user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Target objects:
{target_objects}

Confirmed actions:
{state["confirmed_actions"]}

Confirmed preferences:
{state["confirmed_preferences"]}
{exclusion_block}
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_planned_placements(
            result.placements,
            target_objects=target_objects,
            receptacles=state["receptacles"],
            excluded_receptacles=excluded_receptacles,
        )


def _normalize_planned_placements(
    placements: Dict[str, str],
    *,
    target_objects: List[str],
    receptacles: List[str],
    excluded_receptacles: dict[str, List[str]],
) -> PlacementMap:
    target_set = set(target_objects)
    allowed_receptacles = set(receptacles)
    normalized: PlacementMap = {}
    for obj, receptacle in placements.items():
        if obj not in target_set:
            continue
        if receptacle not in allowed_receptacles:
            continue
        if receptacle in excluded_receptacles.get(obj, []):
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
    finalized = dict(state["predicted_placements_seen"])
    remaining = [obj for obj in state["seen_objects"] if obj not in finalized]
    planned = planner.plan_placements(
        state=state,
        target_objects=remaining,
        scope="seen",
        respect_exclusions=True,
    )
    finalized.update(planned)
    return finalized


def finalize_unseen_placements(
    state: AgentState,
    *,
    planner: FinalPlacementPlanner,
) -> PlacementMap:
    finalized = dict(state["predicted_placements_unseen"])
    remaining = [obj for obj in state["unseen_objects"] if obj not in finalized]
    planned = planner.plan_placements(
        state=state,
        target_objects=remaining,
        scope="unseen",
        respect_exclusions=False,
    )
    finalized.update(planned)
    return finalized


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
    finalized_predicted_seen = finalize_seen_placements(
        state,
        planner=planner,
    )
    finalized_predicted_unseen = finalize_unseen_placements(
        state,
        planner=planner,
    )
    metrics = evaluate_episode_predictions(
        episode,
        predicted_seen=finalized_predicted_seen,
        predicted_unseen=finalized_predicted_unseen,
    )
    return {
        **metrics,
        "finalized_predicted_placements_seen": finalized_predicted_seen,
        "finalized_predicted_placements_unseen": finalized_predicted_unseen,
    }


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
