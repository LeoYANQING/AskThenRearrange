#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import os
import re
from typing import List, Optional, Tuple, Type, Union

from qwen3_api import Qwen3API

from question_generate import (
    generate_question_direct_question,
    generate_question_ParallelExploration,
    generate_question_user_preference,
)
from summarization.benchmark import Scenario, check_placements, load_scenarios


def build_problem_for_remaining(scenario: Scenario, remaining_objects: List[str]) -> str:
    receptacles = ", ".join(scenario.receptacles)
    seen_objects = ", ".join(scenario.seen_objects)
    remaining_str = ", ".join(remaining_objects)
    return (
        f"Room: {scenario.room}\n"
        "Task: Organize the items in the room.\n"
        f"Receptacles: {receptacles}\n"
        f"Seen objects: {seen_objects}\n"
        f"Remaining objects to place: {remaining_str}\n"
        "Review the remaining objects. You can either:\n"
        "1. Ask a question to clarify user preferences or specific item locations (if you are unsure).\n"
        "2. Directly place one or more objects if you can infer their location from history.\n"
        "Choose the most efficient action."
    )


def build_answer_prompt(scenario: Scenario, question: str) -> str:
    receptacles = ", ".join(scenario.receptacles)
    seen_objects = ", ".join(scenario.seen_objects)
    seen_placements = "\n".join(
        [f"- {obj}: {recep}" for obj, recep in scenario.seen_placements]
    )
    return (
        f"Seen placements:\n{seen_placements}\n"
        f"Question: {question}\n"
        "Answer naturally and concisely. If you mention locations, use receptacle names "
        f"from this list: {receptacles}."
    )


def extract_receptacle(answer: str, receptacles: List[str]) -> str:
    normalized = answer.strip().lower()
    for receptacle in receptacles:
        if receptacle.lower() in normalized:
            return receptacle
    for token in normalized.replace(".", " ").replace(",", " ").split():
        for receptacle in receptacles:
            if token == receptacle.lower():
                return receptacle
    return answer.strip()


def extract_question(text: str) -> str:
    cleaned = text.strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_answer(text: str) -> str:
    cleaned = text.strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_question_prompt(problem: str, strategy: str, history: str) -> str:
    normalized = strategy.strip().lower()
    if normalized in {"user_preference", "preference", "k11"}:
        return generate_question_user_preference(problem, history)
    if normalized in {"parallel", "parallel_exploration"}:
        return generate_question_ParallelExploration(problem, history)
    if normalized in {"direct", "direct_question"}:
        return generate_question_direct_question(problem, history)
    raise ValueError(f"Unknown question strategy: {strategy}")


def ensure_question(question: str) -> str:
    if not question:
        return "What should I do next?"
    if "?" in question:
        # Take the last question sentence if multiple appear.
        parts = question.split("?")
        last = parts[-2].strip() if len(parts) >= 2 else question.strip()
        return f"{last}?"
    return f"{question}?"


def load_qwen2_5_api() -> Type["Qwen2_5_7BAPI"]:
    module_name = "qwen2_5_7b_api"
    module_path = os.path.join(os.path.dirname(__file__), "qwen2_5_7B_api.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load qwen2_5_7B_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen2_5_7BAPI


def format_history(history: List[dict]) -> str:
    if not history:
        return "None"
    lines = []
    for item in history:
        lines.append(f"Q: {item['question']}")
        lines.append(f"A: {item['answer']}")
    return "\n".join(lines)


def match_object(name: str, candidates: List[str]) -> Optional[str]:
    name = name.lower().strip()
    for cand in candidates:
        if cand.lower() == name:
            return cand
    # Partial match
    for cand in candidates:
        if cand.lower() in name or name in cand.lower():
            return cand
    return None


def build_dialogue_summary_prompt(
    scenario: Scenario, qa_history: List[dict], placements: List[List[str]]
) -> str:
    history = format_history(qa_history)
    placement_lines = "\n".join([f"- {obj}: {recep}" for obj, recep in placements])
    if not placement_lines:
        placement_lines = "None"
    return (
        "You are summarizing a robot's Q&A with a user about organizing objects.\n"
        "Summarize the user's preferences or placement rules in one concise sentence.\n"
        f"Room: {scenario.room}\n"
        f"Receptacles: {', '.join(scenario.receptacles)}\n"
        f"Q&A history:\n{history}\n"
        f"Known placements:\n{placement_lines}\n"
        "Output only the summary sentence."
    )


def query_placements(
    scenario: Scenario,
    objects: List[str],
    question_model: Optional[Qwen3API] = None,
    answer_model: Optional[Union[Qwen3API, "Qwen2_5_7BAPI"]] = None,
    question_strategy: str = "direct",
) -> Tuple[List[List[str]], List[dict], str]:
    question_model = question_model or Qwen3API()
    if answer_model is None:
        answer_model_class = load_qwen2_5_api()
        answer_model = answer_model_class(model="qwen2.5:7b")

    placements: List[List[str]] = []
    qa_history: List[dict] = []
    
    remaining_objects = list(objects)
    max_turns = len(objects) * 3  # Safety break
    turns = 0

    while remaining_objects and turns < max_turns:
        turns += 1
        problem = build_problem_for_remaining(scenario, remaining_objects)
        history_str = format_history(qa_history)
        question_prompt = build_question_prompt(problem, question_strategy, history_str)
        
        response = question_model.generate(
            question_prompt,
            system="You are a helper robot. Output ONLY the question sentence or 'PLACEMENT: object -> receptacle'. Do NOT output any internal reasoning or analysis.",
            options={"temperature": 0.2, "num_predict": 1024},
        )
        response_text = extract_question(str(response))
        
        # Check for placements
        new_placements = []
        lines = response_text.split('\n')
        is_placement = False
        
        for line in lines:
            if "PLACEMENT:" in line:
                is_placement = True
                # Format: PLACEMENT: object -> receptacle
                content = line.split("PLACEMENT:")[1].strip()
                if "->" in content:
                    obj_part, recep_part = content.split("->", 1)
                    matched_obj = match_object(obj_part.strip(), remaining_objects)
                    parsed_recep = extract_receptacle(recep_part, scenario.receptacles)
                    
                    if matched_obj:
                        new_placements.append((matched_obj, parsed_recep))
        
        if is_placement and new_placements:
            # Apply placements
            for obj, recep in new_placements:
                placements.append([obj, recep])
                if obj in remaining_objects:
                    remaining_objects.remove(obj)
            # We don't ask user anything if we placed items
            # But maybe we should loop again to see if we can place more or need to ask?
            continue
        elif is_placement and not new_placements:
            # Model tried to place but failed to match objects/format
            # Force a question next time or just break to avoid infinite loop?
            # Let's try to ask a fallback question about the first remaining object
            pass

        # If we are here, it means we didn't successfully place anything (or model chose to ask)
        # So we treat response as a question
        question_text = ensure_question(response_text)
        
        # Ask the user
        answer_prompt = build_answer_prompt(scenario, question_text)
        answer = answer_model.generate(
            answer_prompt,
            system="You are the user answering a household organization question.",
            options={"temperature": 0.0, "num_predict": 512},
        )
        answer_text = extract_answer(str(answer))
        
        matched_obj = match_object(question_text, remaining_objects)
        receptacle = extract_receptacle(answer_text, scenario.receptacles)
        if matched_obj and receptacle:
            placements.append([matched_obj, receptacle])
            remaining_objects.remove(matched_obj)

        qa_history.append(
            {
                "question": question_text,
                "answer": answer_text,
                "strategy": question_strategy,
            }
        )
        
    # If we exited loop but still have objects (max turns reached), we might want to log that
    if remaining_objects:
        print(f"Warning: Failed to place {remaining_objects} within turn limit.")

    summary_prompt = build_dialogue_summary_prompt(scenario, qa_history, placements)
    summary = question_model.generate(
        summary_prompt,
        system="Output one concise sentence. No analysis.",
        options={"temperature": 0.0, "num_predict": 128},
    )
    summary_text = extract_answer(str(summary))

    return placements, qa_history, summary_text


def query_seen_placements(
    scenario: Scenario,
    question_model: Optional[Qwen3API] = None,
    answer_model: Optional[Union[Qwen3API, "Qwen2_5_7BAPI"]] = None,
    question_strategy: str = "direct",
) -> Tuple[List[List[str]], List[dict], str]:
    return query_placements(
        scenario,
        scenario.seen_objects,
        question_model=question_model,
        answer_model=answer_model,
        question_strategy=question_strategy,
    )


def evaluate_seen_placements(
    scenarios: List[Scenario],
    question_model: Optional[Qwen3API] = None,
    answer_model: Optional[Union[Qwen3API, "Qwen2_5_7BAPI"]] = None,
    scenario_limit: Optional[int] = None,
    question_strategy: str = "direct",
) -> None:
    question_model = question_model or Qwen3API()
    if answer_model is None:
        answer_model = question_model

    total_correct = 0
    total_items = 0

    for index, scenario in enumerate(scenarios[: scenario_limit or len(scenarios)]):
        placements, qa_history, summary_text = query_seen_placements(
            scenario,
            question_model=question_model,
            answer_model=answer_model,
            question_strategy=question_strategy,
        )
        corrects, accuracy = check_placements(placements, scenario.seen_placements)
        total_correct += sum(corrects)
        total_items += len(corrects)

        print(f"\nScenario {index + 1}: {scenario.room}")
        print("Q&A history:")
        print(json.dumps(qa_history, ensure_ascii=True, indent=2))
        print("\nDialogue summary:")
        print(summary_text)
        print("\nPredicted placements:")
        print(json.dumps(placements, ensure_ascii=True, indent=2))
        print(f"\nScenario accuracy: {accuracy:.2f}")

    overall_accuracy = total_correct / total_items if total_items else 0.0
    print(f"\nOverall accuracy: {overall_accuracy:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate question strategies on seen placements."
    )
    parser.add_argument(
        "--scenarios",
        default=os.path.join("summarization", "scenarios.yml"),
        help="Path to scenarios.yml",
    )
    parser.add_argument(
        "--max_scenarios",
        type=int,
        default=0,
        help="Limit number of scenarios (0 means all).",
    )
    parser.add_argument(
        "--strategies",
        default="direct,user_preference,parallel",
        help="Comma-separated: direct,user_preference,parallel",
    )
    args = parser.parse_args()

    scenarios_path = (
        args.scenarios
        if os.path.isabs(args.scenarios)
        else os.path.join(os.path.dirname(__file__), args.scenarios)
    )
    scenarios = load_scenarios(scenarios_path)
    if not scenarios:
        raise RuntimeError("No scenarios loaded from scenarios.yml")
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    question_model = Qwen3API()
    answer_model = Qwen3API()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} ===")
        evaluate_seen_placements(
            scenarios,
            question_model=question_model,
            answer_model=answer_model,
            question_strategy=strategy,
        )


if __name__ == "__main__":
    main()
