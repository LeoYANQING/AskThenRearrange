"""
Preference-first strategy demo with ChatOllama.with_structured_output.

Flow:
1) Ask a high-level preference question first.
2) Infer which remaining objects are covered by known preferences.
3) Ask action questions for covered objects.
4) If preferences cannot cover remaining objects, ask another preference question.

This script loads a scenario from a dataset JSON file (default:
graph/scenarios_aug_tiny.json) and runs the strategy loop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_ollama import ChatOllama


@dataclass
class Scenario:
    room: str
    receptacles: List[str]
    objects: List[str]
    oracle_notes: List[str]
    oracle_placements: Dict[str, str]


def normalize_placements(placements: object) -> Dict[str, str]:
    if placements is None:
        return {}
    if isinstance(placements, dict):
        return {str(k): str(v) for k, v in placements.items()}
    if isinstance(placements, list):
        out: Dict[str, str] = {}
        for item in placements:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = str(item[1])
        return out
    raise ValueError(f"Unknown placements format: {type(placements)}")


def load_scenarios(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of scenarios.")
    return data


def build_scenario_from_sample(sample: dict) -> Scenario:
    receptacles = [str(x) for x in sample.get("receptacles", [])]
    seen_objects = [str(x) for x in sample.get("seen_objects", [])]
    unseen_objects = [str(x) for x in sample.get("unseen_objects", [])]
    objects = seen_objects + [x for x in unseen_objects if x not in seen_objects]

    seen_placements = normalize_placements(sample.get("seen_placements", {}))
    unseen_placements = normalize_placements(sample.get("unseen_placements", {}))
    oracle_placements = dict(seen_placements)
    oracle_placements.update(unseen_placements)

    for obj in objects:
        if obj not in oracle_placements:
            oracle_placements[obj] = receptacles[0] if receptacles else "unknown"

    return Scenario(
        room=str(sample.get("room", "unknown")),
        receptacles=receptacles,
        objects=objects,
        oracle_notes=[str(x) for x in sample.get("annotator_notes", [])],
        oracle_placements=oracle_placements,
    )


def make_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def _question_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
        },
        "required": ["question"],
        "additionalProperties": False,
    }


def _summary_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "rules": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "rules"],
        "additionalProperties": False,
    }


def _coverage_schema(objects: List[str], receptacles: List[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "covered": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "object": {"type": "string", "enum": objects},
                        "receptacle": {"type": "string", "enum": receptacles},
                        "reason": {"type": "string"},
                    },
                    "required": ["object", "receptacle", "reason"],
                    "additionalProperties": False,
                },
            },
            "uncovered": {
                "type": "array",
                "items": {"type": "string", "enum": objects},
            },
        },
        "required": ["covered", "uncovered"],
        "additionalProperties": False,
    }


def generate_preference_question(
    llm: ChatOllama,
    *,
    method: str,
    scenario: Scenario,
    summary: str,
    unresolved_objects: List[str],
    round_id: int,
) -> str:
    runnable = llm.with_structured_output(_question_schema(), method=method)
    prompt = f"""
You are a household rearrangement agent using preference-first strategy.
Room: {scenario.room}
Receptacles: {scenario.receptacles}
Unresolved objects: {unresolved_objects}
Known preference summary: {summary or "(empty)"}
Preference question round: {round_id}

Write ONE concise high-level preference question.
It should discover organizing principles that can help place multiple unresolved objects.
""".strip()
    result = runnable.invoke([{"role": "user", "content": prompt}])
    return result["question"].strip()


def update_preference_summary(
    llm: ChatOllama,
    *,
    method: str,
    previous_summary: str,
    qa_history: List[Dict[str, str]],
) -> Tuple[str, List[str]]:
    runnable = llm.with_structured_output(_summary_schema(), method=method)
    prompt = f"""
Update preference summary from conversation.

Previous summary:
{previous_summary or "(empty)"}

Preference QA history:
{qa_history}

Return a compact summary and a list of explicit rules.
""".strip()
    result = runnable.invoke([{"role": "user", "content": prompt}])
    summary = result["summary"].strip()
    rules = [x.strip() for x in result["rules"] if isinstance(x, str) and x.strip()]
    return summary, rules


def infer_coverage(
    llm: ChatOllama,
    *,
    method: str,
    scenario: Scenario,
    summary: str,
    rules: List[str],
    unresolved_objects: List[str],
) -> Dict[str, str]:
    if not unresolved_objects:
        return {}

    runnable = llm.with_structured_output(
        _coverage_schema(unresolved_objects, scenario.receptacles), method=method
    )
    prompt = f"""
Given known user preferences, decide which unresolved objects can be placed confidently.

Room: {scenario.room}
Receptacles: {scenario.receptacles}
Unresolved objects: {unresolved_objects}

Preference summary:
{summary or "(empty)"}

Rules:
{rules}

Output covered objects only when confidence is high.
""".strip()
    result = runnable.invoke([{"role": "user", "content": prompt}])

    covered_map: Dict[str, str] = {}
    for item in result.get("covered", []):
        if not isinstance(item, dict):
            continue
        obj = item.get("object")
        rec = item.get("receptacle")
        if isinstance(obj, str) and isinstance(rec, str) and obj in unresolved_objects:
            covered_map[obj] = rec
    return covered_map


def generate_action_question(
    llm: ChatOllama,
    *,
    method: str,
    scenario: Scenario,
    summary: str,
    target_object: str,
    suggested_receptacle: str,
) -> str:
    runnable = llm.with_structured_output(_question_schema(), method=method)
    prompt = f"""
You are a household rearrangement agent.
Known preference summary:
{summary}

Target object: {target_object}
Likely receptacle from preferences: {suggested_receptacle}
Allowed receptacles: {scenario.receptacles}

Write ONE concise action question asking where to place this object.
""".strip()
    result = runnable.invoke([{"role": "user", "content": prompt}])
    return result["question"].strip()


def oracle_answer_preference(scenario: Scenario, preference_round: int) -> str:
    idx = min(preference_round - 1, len(scenario.oracle_notes) - 1)
    return scenario.oracle_notes[idx]


def oracle_answer_action(scenario: Scenario, target_object: str) -> str:
    receptacle = scenario.oracle_placements[target_object]
    return f"Place {target_object} in {receptacle}."


def run_preference_first_demo(
    llm: ChatOllama,
    *,
    scenario: Scenario,
    method: str,
    max_turns: int,
) -> None:
    unresolved = list(scenario.objects)
    resolved: Dict[str, str] = {}
    preference_summary = ""
    preference_rules: List[str] = []
    preference_history: List[Dict[str, str]] = []
    preference_round = 0
    turns = 0

    print("=== Preference-first structured-output demo ===")
    print(f"Room: {scenario.room}")
    print(f"Objects: {scenario.objects}")
    print(f"Receptacles: {scenario.receptacles}")

    while unresolved and turns < max_turns:
        covered_map = infer_coverage(
            llm,
            method=method,
            scenario=scenario,
            summary=preference_summary,
            rules=preference_rules,
            unresolved_objects=unresolved,
        )

        should_ask_preference = (not preference_summary) or (len(covered_map) == 0)

        if should_ask_preference:
            turns += 1
            preference_round += 1
            q = generate_preference_question(
                llm,
                method=method,
                scenario=scenario,
                summary=preference_summary,
                unresolved_objects=unresolved,
                round_id=preference_round,
            )
            a = oracle_answer_preference(scenario, preference_round)
            preference_history.append({"question": q, "answer": a})
            preference_summary, preference_rules = update_preference_summary(
                llm,
                method=method,
                previous_summary=preference_summary,
                qa_history=preference_history,
            )
            print(f"\n[Turn {turns}] preference")
            print("Q:", q)
            print("A:", a)
            print("Updated summary:", preference_summary)
            continue

        covered_targets = [obj for obj in unresolved if obj in covered_map]
        if not covered_targets:
            continue

        for target in covered_targets:
            if turns >= max_turns:
                break
            turns += 1
            q = generate_action_question(
                llm,
                method=method,
                scenario=scenario,
                summary=preference_summary,
                target_object=target,
                suggested_receptacle=covered_map[target],
            )
            a = oracle_answer_action(scenario, target)
            resolved[target] = scenario.oracle_placements[target]
            unresolved = [x for x in unresolved if x != target]
            print(f"\n[Turn {turns}] action")
            print("Q:", q)
            print("A:", a)
            print(f"Resolved: {target} -> {resolved[target]}")

    print("\n=== Final state ===")
    print("Resolved placements:", json.dumps(resolved, ensure_ascii=False, indent=2))
    print("Remaining unresolved:", unresolved)
    print("Final preference summary:", preference_summary)
    if unresolved:
        print("Stopped because max_turns reached before full coverage.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preference-first strategy demo using with_structured_output."
    )
    parser.add_argument("--model", default="qwen3.5", help="Ollama model name.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--temperature", type=float, default=0, help="Model temperature.")
    parser.add_argument(
        "--method",
        default="json_schema",
        choices=["json_schema", "json_mode", "function_calling"],
        help="Structured output method.",
    )
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum dialog turns.")
    parser.add_argument(
        "--dataset",
        default="graph/scenarios_aug_tiny.json",
        help="Path to dataset JSON.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Scenario index in dataset.",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.dataset)
    if args.index < 0 or args.index >= len(scenarios):
        raise IndexError(f"--index out of range: {args.index}, valid 0..{len(scenarios)-1}")
    scenario = build_scenario_from_sample(scenarios[args.index])

    llm = make_llm(args.model, args.base_url, args.temperature)
    run_preference_first_demo(
        llm,
        scenario=scenario,
        method=args.method,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
