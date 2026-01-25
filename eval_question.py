#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from qwen2_5_7B_api import Qwen2_5_7BAPI
from qwen3_api import Qwen3API
from task_matter import query_seen_placements
from summarization import benchmark, utils
from summarization.openai_cache import Completion


def construct_summarization_prompt(objects, receptacles, placements) -> str:
    summarization_prompt_template = '''objects = ["dried figs", "protein bar", "cornmeal", "Macadamia nuts", "vinegar", "herbal tea", "peanut oil", "chocolate bar", "bread crumbs", "Folgers instant coffee"]
receptacles = ["top rack", "middle rack", "table", "shelf", "plastic box"]
pick_and_place("dried figs", "plastic box")
pick_and_place("protein bar", "shelf")
pick_and_place("cornmeal", "top rack")
pick_and_place("Macadamia nuts", "plastic box")
pick_and_place("vinegar", "middle rack")
pick_and_place("herbal tea", "table")
pick_and_place("peanut oil", "middle rack")
pick_and_place("chocolate bar", "shelf")
pick_and_place("bread crumbs", "top rack")
pick_and_place("Folgers instant coffee", "table")
# Summary: Put dry ingredients on the top rack, liquid ingredients in the middle rack, tea and coffee on the table, packaged snacks on the shelf, and dried fruits and nuts in the plastic box.

objects = ["yoga pants", "wool sweater", "black jeans", "Nike shorts"]
receptacles = ["hamper", "bed"]
pick_and_place("yoga pants", "hamper")
pick_and_place("wool sweater", "bed")
pick_and_place("black jeans", "bed")
pick_and_place("Nike shorts", "hamper")
# Summary: Put athletic clothes in the hamper and other clothes on the bed.

objects = ["Nike sweatpants", "sweater", "cargo shorts", "iPhone", "dictionary", "tablet", "Under Armour t-shirt", "physics homework"]
receptacles = ["backpack", "closet", "desk", "nightstand"]
pick_and_place("Nike sweatpants", "backpack")
pick_and_place("sweater", "closet")
pick_and_place("cargo shorts", "closet")
pick_and_place("iPhone", "nightstand")
pick_and_place("dictionary", "desk")
pick_and_place("tablet", "nightstand")
pick_and_place("Under Armour t-shirt", "backpack")
pick_and_place("physics homework", "desk")
# Summary: Put workout clothes in the backpack, other clothes in the closet, books and homeworks on the desk, and electronics on the nightstand.

objects = {objects_str}
receptacles = {receptacles_str}
{placements_str}
# Summary:'''
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    placements_str = '\n'.join(map(lambda x: f'pick_and_place("{x[0]}", "{x[1]}")', placements))
    return summarization_prompt_template.format(
        objects_str=objects_str,
        receptacles_str=receptacles_str,
        placements_str=placements_str,
    )


def construct_placement_prompt(summary, objects, receptacles) -> str:
    placement_prompt_template = '''# Summary: Put clothes in the laundry basket and toys in the storage box.
objects = ["socks", "toy car", "shirt", "Lego brick"]
receptacles = ["laundry basket", "storage box"]
pick_and_place("socks", "laundry basket")
pick_and_place("toy car", "storage box")
pick_and_place("shirt", "laundry basket")
pick_and_place("Lego brick", "storage box")

# Summary: {summary}
objects = {objects_str}
receptacles = {receptacles_str}
pick_and_place("{first_object}",'''
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    return placement_prompt_template.format(
        summary=summary,
        objects_str=objects_str,
        receptacles_str=receptacles_str,
        first_object=objects[0],
    )


def strip_think(text: str) -> str:
    cleaned = text.strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def clean_summary(text: str) -> str:
    cleaned = strip_think(text)
    summary_match = re.search(r"summary:\s*(.+)", cleaned, flags=re.IGNORECASE)
    if summary_match:
        return summary_match.group(1).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0]


def normalize_name(text: str) -> str:
    return (
        text.strip()
        .replace('"', "")
        .replace("'", "")
        .replace(")", "")
        .replace("(", "")
        .strip()
        .lower()
    )


class ModelRunner:
    def __init__(self, model_name: str) -> None:
        normalized = model_name.strip()
        lower_name = normalized.lower()
        if lower_name.startswith("qwen3"):
            model = normalized if ":" in normalized else "qwen3:32b"
            self.client = Qwen3API(model=model)
            self.backend = "qwen3"
        elif lower_name.startswith("qwen2.5") or lower_name.startswith("qwen2_5"):
            model = normalized if ":" in normalized else "qwen2.5:7b"
            self.client = Qwen2_5_7BAPI(model=model)
            self.backend = "qwen2.5"
        else:
            self.client = Completion()
            self.backend = "openai"
        self.model_name = normalized

    def generate(self, prompt: str) -> str:
        if self.backend in {"qwen3", "qwen2.5"}:
            response = self.client.generate(
                prompt,
                options={"temperature": 0.0, "num_predict": 256},
            )
            return str(response).strip()
        response = self.client.create(prompt, model=self.model_name)
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, "choices") and response.choices:
            return str(response.choices[0].message.content).strip()
        return str(response).strip()


def parse_model_placements(
    text: str,
    objects: List[str],
    receptacles: List[str],
) -> List[List[str]]:
    cleaned = strip_think(text)
    object_lookup = {normalize_name(obj): obj for obj in objects}
    receptacle_lookup = {normalize_name(recep): recep for recep in receptacles}
    placements: List[List[str]] = []
    first_object = objects[0] if objects else None
    first_assigned = False
    pattern = re.compile(
        r'pick_and_place\(\s*["\']?(?P<obj>[^,"\')]+?)["\']?\s*,\s*["\']?(?P<rec>[^"\')]+?)["\']?\s*\)'
    )

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = pattern.search(line)
        if match:
            obj = object_lookup.get(normalize_name(match.group("obj")))
            rec = receptacle_lookup.get(normalize_name(match.group("rec")))
            if obj and rec:
                placements.append([obj, rec])
                if obj == first_object:
                    first_assigned = True
            continue

        if "," in line:
            left, right = line.split(",", 1)
            obj = object_lookup.get(normalize_name(left))
            rec = receptacle_lookup.get(normalize_name(right))
            if obj and rec:
                placements.append([obj, rec])
                if obj == first_object:
                    first_assigned = True
            continue

        rec = receptacle_lookup.get(normalize_name(line))
        if rec and first_object and not first_assigned:
            placements.append([first_object, rec])
            first_assigned = True

    if placements:
        return placements
    return benchmark.parse_placements(cleaned, objects)


def evaluate_questions(
    scenarios,
    eval_split: str = "unseen",
    model_name: str = "gpt-4o",
    verbose: bool = False,
    use_dialogue: bool = True,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    # 功能：评估 LLM 总结整理规则和根据总结预测物体放置的能力。
    assert eval_split in {"unseen", "seen"}
    runner = ModelRunner(model_name)
    accuracies: List[float] = []
    details: List[Dict[str, Any]] = []

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"Scenario {i + 1} of {len(scenarios)}\n")

        seen_result = query_seen_placements(scenario)
        if isinstance(seen_result, tuple) and len(seen_result) == 2:
            seen_placements, qa_history = seen_result
        else:
            seen_placements, qa_history = seen_result, []

        summarization_prompt = construct_summarization_prompt(
            scenario.seen_objects, scenario.receptacles, seen_placements
        )
        summarization_completion = runner.generate(summarization_prompt)
        if verbose:
            print(summarization_prompt, end="")
            utils.print_colored(summarization_completion, "blue")
            print("\n" + 10 * "-" + "\n")

        summary = clean_summary(summarization_completion)
        objects = scenario.seen_objects if eval_split == "seen" else scenario.unseen_objects
        correct_placements = (
            scenario.seen_placements if eval_split == "seen" else scenario.unseen_placements
        )

        placement_prompt = ""
        placement_completion = ""
        predicted_placements: List[List[str]] = []
        corrects: List[bool] = []
        if objects and correct_placements:
            placement_prompt = construct_placement_prompt(
                summary, objects, scenario.receptacles
            )
            placement_completion = runner.generate(placement_prompt)
            if verbose:
                print(placement_prompt, end="")
                utils.print_colored(placement_completion, "blue")
                print("\n" + 10 * "-" + "\n")

            predicted_placements = parse_model_placements(
                placement_completion, objects, scenario.receptacles
            )
            corrects, accuracy = benchmark.check_placements(
                predicted_placements, correct_placements
            )
        else:
            accuracy = 1.0 if not correct_placements else 0.0

        accuracies.append(accuracy)
        detail: Dict[str, Any] = {
            "room": scenario.room,
            "summary": summary,
            "accuracy": accuracy,
            "correct_placements": correct_placements,
            "predicted_placements": predicted_placements,
        }
        if use_dialogue:
            detail["qa_history"] = qa_history
        details.append(detail)

        if verbose:
            print(f"Annotator notes: {scenario.annotator_notes}\n")
            print("Correct placements:")
            for placement in correct_placements:
                print(placement)
            print("\nParsed placements:")
            for placement, correct in zip(predicted_placements, corrects):
                utils.print_colored(placement, "green" if correct else "red")
            print(f"\nAccuracy: {accuracy:.2f}")
            print("\n" + 80 * "-" + "\n")
    return accuracies, details

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate question-based placement accuracy on seen objects."
    )
    parser.add_argument(
        "--scenarios",
        default=os.path.join("summarization", "scenarios.yml"),
        help="Path to scenarios.yml",
    )
    parser.add_argument(
        "--models",
        default="qwen3",
        help="Comma-separated list: qwen3,qwen2.5",
    )
    parser.add_argument(
        "--max_scenarios",
        type=int,
        default=0,
        help="Limit number of scenarios (0 means all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-scenario details.",
    )
    args = parser.parse_args()

    scenarios = benchmark.load_scenarios(args.scenarios)
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    model_list = [name.strip() for name in args.models.split(",") if name.strip()]
    results = {}
    for model_name in model_list:
        seen_accuracies, seen_details = evaluate_questions(
            scenarios, eval_split="seen", model_name=model_name, verbose=args.verbose
        )
        unseen_accuracies, unseen_details = evaluate_questions(
            scenarios, eval_split="unseen", model_name=model_name, verbose=args.verbose
        )

        results[model_name] = {
            "seen": {
                "accuracies": seen_accuracies,
                "average": sum(seen_accuracies) / len(seen_accuracies)
                if seen_accuracies
                else 0.0,
                "details": seen_details,
            },
            "unseen": {
                "accuracies": unseen_accuracies,
                "average": sum(unseen_accuracies) / len(unseen_accuracies)
                if unseen_accuracies
                else 0.0,
                "details": unseen_details,
            },
        }

    os.makedirs("result", exist_ok=True)
    output_path = os.path.join("result", "eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    for model_name in model_list:
        model_result = results.get(model_name, {})
        seen_avg = model_result.get("seen", {}).get("average", 0.0)
        unseen_avg = model_result.get("unseen", {}).get("average", 0.0)
        print(f"{model_name} Seen accuracy: {seen_avg:.2f}")
        print(f"{model_name} Unseen accuracy: {unseen_avg:.2f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
