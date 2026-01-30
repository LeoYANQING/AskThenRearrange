#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ollama_call import VLMAPI
from benchmark.organizer import Organizer, parse_fact

from eval_question import (
    ModelRunner,
    clean_summary,
    construct_summarization_prompt,
    evaluate_questions,
    plot_accuracy_curve,
)
STRICT_RULES = (
    "Strict rules:\n"
    "* Do not include any internal reasoning, thinking, or analysis.\n"
    "* Only return the final answer.\n"
)

class QuestionOutput(BaseModel):
    question: str

class AnswerOutput(BaseModel):
    answer: str

TModel = TypeVar("TModel", bound=BaseModel)

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from question_generate import (
    generate_question_direct_question,
    generate_question_ParallelExploration,
    generate_question_general,
    generate_question_user_preference,
)
from summarization import benchmark
from summarization.benchmark import Scenario


def build_problem_for_seen(scenario: Scenario) -> str:
    receptacles = ", ".join(scenario.receptacles)
    seen_objects = ", ".join(scenario.seen_objects)
    return (
        STRICT_RULES
        + "* Output format: JSON that matches the provided schema.\n"
        f"Room: {scenario.room}\n"
        "Task: Organize the items in the room.\n"
        f"Receptacles: {receptacles}\n"
        f"Seen objects: {seen_objects}\n"
        "Ask one clear question to clarify user preferences or specific item locations.\n"
        "Only return the question field in JSON."
    )


def build_answer_prompt(scenario: Scenario, question: str) -> str:
    receptacles = ", ".join(scenario.receptacles)
    seen_objects = ", ".join(scenario.seen_objects)
    seen_placements = "\n".join(
        [f"- {obj}: {recep}" for obj, recep in scenario.seen_placements]
    )
    return (
        STRICT_RULES + 
        f"Seen placements:\n{seen_placements}\n"
        f"Question: {question}\n"
        "Answer naturally and concisely. If you mention locations, use receptacle names "
        f"from this list: {receptacles}. Return only the answer content."
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


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_structured_output(text: str, model: Type[TModel]) -> Optional[TModel]:
    cleaned = text.strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = _strip_code_fence(cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    try:
        return model.model_validate(data)
    except ValidationError:
        return None


class QuestionPromptBuilder:
    _ALIASES = {
        "general": "general",
        "user_preference": "user_preference",
        "preference": "user_preference",
        "k11": "user_preference",
        "parallel": "parallel",
        "parallel_exploration": "parallel",
        "direct": "direct",
        "direct_question": "direct",
    }
    _STRATEGIES = {
        "general": generate_question_general,
        "user_preference": generate_question_user_preference,
        "parallel": generate_question_ParallelExploration,
        "direct": generate_question_direct_question,
    }

    def __init__(self, strategy: str = "direct") -> None:
        self.strategy = "direct"
        self.set_strategy(strategy)

    @classmethod
    def normalize_strategy(cls, strategy: str) -> str:
        normalized = strategy.strip().lower()
        if normalized in cls._ALIASES:
            return cls._ALIASES[normalized]
        return normalized

    @classmethod
    def available_strategies(cls) -> List[str]:
        return sorted(cls._STRATEGIES.keys())

    def set_strategy(self, strategy: str) -> None:
        normalized = self.normalize_strategy(strategy)
        if normalized not in self._STRATEGIES:
            raise ValueError(f"Unknown question strategy: {strategy}")
        self.strategy = normalized

    def build(self, problem: str, history: str) -> str:
        builder = self._STRATEGIES[self.strategy]
        return builder(problem, history)

    def __call__(self, problem: str, history: str) -> str:
        return self.build(problem, history)


def build_question_prompt(problem: str, strategy: str, history: str) -> str:
    return QuestionPromptBuilder(strategy).build(problem, history)


def ensure_question(question: str) -> str:
    if not question:
        return "What should I do next?"
    if "?" in question:
        # Take the last question sentence if multiple appear.
        parts = question.split("?")
        last = parts[-2].strip() if len(parts) >= 2 else question.strip()
        return f"{last}?"
    return f"{question}?"


def resolve_max_questions(objects: List[str], max_questions: Optional[int]) -> int:
    if max_questions is None or max_questions <= 0:
        return len(objects) * 3
    return max_questions


def resolve_log_path(
    log_path: Optional[str],
    strategy: str,
    max_questions: Optional[int],
) -> str:
    if log_path:
        if "{strategy}" in log_path or "{max_questions}" in log_path:
            return log_path.format(
                strategy=strategy, max_questions=max_questions or "auto"
            )
        return log_path
    safe_strategy = re.sub(r"[^a-zA-Z0-9_-]+", "_", strategy)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "summarization", "test_logs")
    filename = f"test_log_{safe_strategy}_q{max_questions or 'auto'}_{timestamp}.json"
    return os.path.join(base_dir, filename)


def resolve_strategy_log_path(
    log_path: Optional[str],
    strategy: str,
    max_questions: Optional[int],
    parallel: bool,
    strategy_count: int,
) -> str:
    if log_path:
        if not os.path.isabs(log_path):
            log_path = os.path.join(os.path.dirname(__file__), log_path)
        if parallel and strategy_count > 1 and "{strategy}" not in log_path:
            safe_strategy = re.sub(r"[^a-zA-Z0-9_-]+", "_", strategy)
            root, ext = os.path.splitext(log_path)
            ext = ext or ".json"
            log_path = f"{root}_{safe_strategy}{ext}"
        return resolve_log_path(log_path, strategy, max_questions)
    return resolve_log_path(None, strategy, max_questions)


def resolve_strategy_workers(strategy_workers: Optional[int], strategy_count: int) -> int:
    if strategy_count <= 0:
        return 1
    if not strategy_workers or strategy_workers <= 0:
        return min(32, strategy_count)
    return max(1, min(strategy_workers, strategy_count))


def append_test_log(log_path: str, entry: Dict[str, object]) -> None:
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_scenarios_json(path: str) -> List[Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "scenarios" in data and isinstance(data["scenarios"], list):
            items = data["scenarios"]
        else:
            items = [data]
    else:
        raise ValueError("Unsupported JSON format for scenarios.")

    required_keys = {
        "annotator_notes",
        "receptacles",
        "room",
        "seen_objects",
        "seen_placements",
        "tags",
        "unseen_objects",
        "unseen_placements",
    }
    scenarios: List[Scenario] = []
    for i, item in enumerate(items):
        missing = required_keys - set(item.keys())
        if missing:
            raise KeyError(f"Scenario {i} missing keys: {missing}")
        scenarios.append(Scenario(**item))
    return scenarios


def _call_vlm(
    llm: VLMAPI,
    prompt: str,
    system: str,
    temperature: float,
    format_schema: Optional[dict] = None,
) -> str:
    return llm.vlm_request_with_format(
        system,
        prompt,
        format_schema=format_schema,
        options={"temperature": temperature},
    )


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


def placements_to_objects(placements: List[List[str]]) -> List[Dict[str, str]]:
    return [{"object": obj, "placement": recep} for obj, recep in placements]


def _normalize_id(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    return normalized.strip("_") or "item"


def build_id_maps(names: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    for name in names:
        base = _normalize_id(name)
        candidate = base
        counter = 2
        while candidate in id_to_name and id_to_name[candidate] != name:
            candidate = f"{base}_{counter}"
            counter += 1
        name_to_id[name] = candidate
        id_to_name[candidate] = name
    return name_to_id, id_to_name


def build_organizer_sample(
    scenario: Scenario,
    qa_history: List[dict],
) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, str]]:
    obj_name_to_id, obj_id_to_name = build_id_maps(scenario.seen_objects)
    rec_name_to_id, rec_id_to_name = build_id_maps(scenario.receptacles)
    objects = {obj_id: obj_id for obj_id in obj_id_to_name.keys()}
    receptacles = {rec_id: rec_id for rec_id in rec_id_to_name.keys()}
    init: List[str] = []
    history = [
        {
            "turn_id": idx + 1,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        }
        for idx, item in enumerate(qa_history)
    ]
    sample = {
        "task": f"Room: {scenario.room}. Organize the items in the room.",
        "objects": objects,
        "receptacles": receptacles,
        "init": init,
        "qa_history": history,
    }
    return sample, obj_id_to_name, rec_id_to_name


def pred_goal_to_placements(
    pred_goal: List[str],
    obj_id_to_name: Dict[str, str],
    rec_id_to_name: Dict[str, str],
) -> List[List[str]]:
    placements: List[List[str]] = []
    for fact in pred_goal:
        pred, obj, receptacle = parse_fact(fact)
        if pred and obj and receptacle:
            placements.append(
                [
                    obj_id_to_name.get(obj, obj),
                    rec_id_to_name.get(receptacle, receptacle),
                ]
            )
    return placements


def query_placements(
    scenario: Scenario,
    objects: List[str],
    question_model: Optional[VLMAPI] = None,
    answer_model: Optional[VLMAPI] = None,
    question_strategy: str = "direct",
    question_builder: Optional[QuestionPromptBuilder] = None,
    max_questions: Optional[int] = None,
) -> List[dict]:
    question_model = question_model or VLMAPI("qwen3:32b")
    answer_model = answer_model or VLMAPI("qwen3:32b")
    question_builder = question_builder or QuestionPromptBuilder(question_strategy)
    question_builder.set_strategy(question_strategy)

    qa_history: List[dict] = []

    required_questions = resolve_max_questions(objects, max_questions)
    questions_asked = 0

    while questions_asked < required_questions:
        problem = build_problem_for_seen(scenario)
        history_str = format_history(qa_history)
        question_prompt = question_builder.build(problem, history_str)
        
        response_raw = _call_vlm(
            question_model,
            question_prompt,
            STRICT_RULES
            + "* Output must be valid JSON matching the provided schema.\n"
            "You are a helper robot. Ask one clear question about the remaining objects.",
            temperature=0.2,
            format_schema=QuestionOutput.model_json_schema(),
        )

        structured_question = parse_structured_output(response_raw, QuestionOutput)
        response_text = (
            structured_question.question
            if structured_question
            else extract_question(response_raw)
        )

        # We always ask a question (no placement decision here).
        question_text = ensure_question(response_text)
        
        # Ask the user
        answer_prompt = build_answer_prompt(scenario, question_text)
        answer_raw = _call_vlm(
            answer_model,
            answer_prompt,
            "You are the user answering a household organization question. "
            "Output only JSON that matches the provided schema. Do NOT include reasoning or analysis.",
            temperature=0.0,
            format_schema=AnswerOutput.model_json_schema(),
        )
        structured_answer = parse_structured_output(answer_raw, AnswerOutput)
        answer_text = (
            structured_answer.answer if structured_answer else extract_answer(answer_raw)
        )
        
        qa_history.append(
            {
                "question": question_text,
                "answer": answer_text,
                "strategy": question_strategy,
            }
        )
        questions_asked += 1

    return qa_history


def run_strategy_scenarios(
    scenarios: List[Scenario],
    strategy: str,
    max_questions: Optional[int],
    log_path: str,
    eval_model: str,
    question_model: Optional[VLMAPI] = None,
    answer_model: Optional[VLMAPI] = None,
    verbose: bool = True,
) -> Tuple[List[List[dict]], List[float]]:
    question_model = question_model or VLMAPI("qwen3:32b")
    answer_model = answer_model or question_model
    question_builder = QuestionPromptBuilder(strategy)
    eval_runner = ModelRunner(eval_model)
    organizer = Organizer(question_model.model)

    output_chunks: List[str] = []
    qa_histories: List[List[dict]] = []
    accuracies: List[float] = []
    for index, scenario in enumerate(scenarios):
        qa_history = query_placements(
            scenario,
            scenario.seen_objects,
            question_model=question_model,
            answer_model=answer_model,
            question_strategy=strategy,
            question_builder=question_builder,
            max_questions=max_questions,
        )
        qa_histories.append(qa_history)

        summary_prompt = construct_summarization_prompt(
            scenario.seen_objects,
            scenario.receptacles,
            qa_history=qa_history,
        )
        summary_completion = eval_runner.generate(summary_prompt)
        summary = clean_summary(summary_completion)
        organizer_sample, obj_id_to_name, rec_id_to_name = build_organizer_sample(
            scenario, qa_history
        )
        pred_goal = organizer.predict_goal(organizer_sample)
        predicted_placements = pred_goal_to_placements(
            pred_goal, obj_id_to_name, rec_id_to_name
        )
        _corrects, accuracy = benchmark.check_placements(
            predicted_placements, scenario.seen_placements
        )
        accuracies.append(accuracy)
        predicted_placement_objects = placements_to_objects(predicted_placements)

        log_entry = {"scenario_index": index + 1, **asdict(scenario)}
        log_entry.update(
            {
                "strategy": strategy,
                "max_questions": resolve_max_questions(
                    scenario.seen_objects, max_questions
                ),
                "questions_asked": len(qa_history),
                "qa_history": qa_history,
                "summary": summary,
                "predicted_placements": predicted_placement_objects,
                "accuracy": accuracy,
            }
        )
        append_test_log(log_path, log_entry)

        if verbose:
            output_chunks.append(f"\nScenario {index + 1} (raw):")
            output_chunks.append(
                json.dumps(asdict(scenario), ensure_ascii=True, indent=2)
            )
            output_chunks.append("\nQ&A history:")
            output_chunks.append(json.dumps(qa_history, ensure_ascii=True, indent=2))
            output_chunks.append("\nSummary:")
            output_chunks.append(summary)
            output_chunks.append("\nPredicted placements:")
            output_chunks.append(
                json.dumps(predicted_placement_objects, ensure_ascii=True, indent=2)
            )
            output_chunks.append(f"\nScenario accuracy: {accuracy:.2f}")

    if verbose and output_chunks:
        print("\n".join(output_chunks))

    return qa_histories, accuracies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate question strategies on seen placements (JSON input)."
    )
    parser.add_argument(
        "--scenarios",
        default=os.path.join("summarization", "scenarios_aug.json"),
        help="Path to scenarios JSON.",
    )
    parser.add_argument(
        "--max_scenarios",
        type=int,
        default=0,
        help="Limit number of scenarios (0 means all).",
    )
    parser.add_argument(
        "--strategies",
        default="direct,user_preference,parallel,general",
        help="Comma-separated: direct,user_preference,parallel,general",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Required question turns per scenario (0 means auto).",
    )
    parser.add_argument(
        "--max_questions_list",
        default="1,3,5,8,10",
        help="Comma-separated max question counts to run.",
    )
    parser.add_argument(
        "--log_path",
        default="",
        help="Optional JSON log path (supports {strategy} and {max_questions}).",
    )
    parser.add_argument(
        "--parallel_strategies",
        action="store_true",
        help="Run different strategies in parallel.",
    )
    parser.add_argument(
        "--strategy_workers",
        type=int,
        default=0,
        help="Max worker threads for parallel strategies (0 means auto).",
    )
    parser.add_argument(
        "--eval_model",
        default="qwen3",
        help="Model name for summarization/placement evaluation.",
    )
    parser.add_argument(
        "--plot_path",
        default=os.path.join("summarization", "acc_curve.png"),
        help="Output path for accuracy curve plot.",
    )
    args = parser.parse_args()

    scenarios_path = (
        args.scenarios
        if os.path.isabs(args.scenarios)
        else os.path.join(os.path.dirname(__file__), args.scenarios)
    )
    scenarios = load_scenarios_json(scenarios_path)
    if not scenarios:
        raise RuntimeError("No scenarios loaded from JSON")
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.max_questions > 0:
        max_questions_list = [args.max_questions]
    else:
        max_questions_list = [
            int(v) for v in args.max_questions_list.split(",") if v.strip()
        ]
    overall_start = time.time()
    results_by_strategy = {strategy: [] for strategy in strategies}
    for max_questions in max_questions_list:
        print(f"\n=== Max questions: {max_questions} ===")
        if args.parallel_strategies and len(strategies) > 1:
            max_workers = resolve_strategy_workers(args.strategy_workers, len(strategies))
            print(f"Running strategies in parallel (workers={max_workers})")

            def run_strategy(
                strategy: str,
            ) -> Tuple[str, str, float, List[List[dict]], List[float]]:
                strategy_start = time.time()
                resolved_log_path = resolve_strategy_log_path(
                    args.log_path or None,
                    strategy,
                    max_questions,
                    parallel=True,
                    strategy_count=len(strategies),
                )
                qa_histories, accuracies = run_strategy_scenarios(
                    scenarios,
                    strategy,
                    max_questions,
                    resolved_log_path,
                    args.eval_model,
                    question_model=None,
                    answer_model=None,
                    verbose=True,
                )
                elapsed = time.time() - strategy_start
                return strategy, resolved_log_path, elapsed, qa_histories, accuracies

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_strategy, s): s for s in strategies}
                for future in as_completed(futures):
                    strategy, log_path, elapsed, _qa_histories, accuracies = (
                        future.result()
                    )
                    avg_accuracy = (
                        sum(accuracies) / len(accuracies) if accuracies else 0.0
                    )
                    results_by_strategy[strategy].append(avg_accuracy)
                    print(f"\n=== Strategy: {strategy} ===")
                    print(f"Log saved to: {log_path}")
                    print(f"Strategy time: {elapsed:.2f}s")
                    print(f"Seen accuracy: {avg_accuracy:.2f}")
        else:
            question_model = VLMAPI("qwen3:32b")
            answer_model = VLMAPI("qwen3:32b")
            for strategy in strategies:
                strategy_start = time.time()
                print(f"\n=== Strategy: {strategy} ===")
                log_path = resolve_strategy_log_path(
                    args.log_path or None,
                    strategy,
                    max_questions,
                    parallel=False,
                    strategy_count=len(strategies),
                )
                qa_histories, accuracies = run_strategy_scenarios(
                    scenarios,
                    strategy,
                    max_questions,
                    log_path,
                    args.eval_model,
                    question_model=question_model,
                    answer_model=answer_model,
                    verbose=True,
                )
                avg_accuracy = (
                    sum(accuracies) / len(accuracies) if accuracies else 0.0
                )
                results_by_strategy[strategy].append(avg_accuracy)
                elapsed = time.time() - strategy_start
                print(f"Strategy time: {elapsed:.2f}s")
                print(f"Seen accuracy: {avg_accuracy:.2f}")

    total_elapsed = time.time() - overall_start
    print(f"\nTotal runtime: {total_elapsed:.2f}s")
    plot_accuracy_curve(max_questions_list, results_by_strategy, args.plot_path)
    print(f"Saved accuracy curve to {args.plot_path}")


if __name__ == "__main__":
    main()
