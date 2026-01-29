#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ollama_call import VLMAPI

STRICT_RULES = (
    "Strict rules:\n"
    "* Do not include any internal reasoning, thinking, or analysis.\n"
    "* Only return the final answer.\n"
)


class QuestionOutput(BaseModel):
    question: str


class AnswerOutput(BaseModel):
    answer: str


class SummaryOutput(BaseModel):
    summary: str


TModel = TypeVar("TModel", bound=BaseModel)

from question_generate import (
    generate_question_direct_question,
    generate_question_ParallelExploration,
    generate_question_user_preference,
)
from summarization.benchmark import Scenario, check_placements


def build_problem_for_remaining(scenario: Scenario, remaining_objects: List[str]) -> str:
    receptacles = ", ".join(scenario.receptacles)
    seen_objects = ", ".join(scenario.seen_objects)
    remaining_str = ", ".join(remaining_objects)
    return (
        STRICT_RULES
        + "* Output format: JSON that matches the provided schema.\n"
        f"Room: {scenario.room}\n"
        "Task: Organize the items in the room.\n"
        f"Receptacles: {receptacles}\n"
        f"Seen objects: {seen_objects}\n"
        f"Remaining objects to place: {remaining_str}\n"
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
        "user_preference": "user_preference",
        "preference": "user_preference",
        "k11": "user_preference",
        "parallel": "parallel",
        "parallel_exploration": "parallel",
        "direct": "direct",
        "direct_question": "direct",
    }
    _STRATEGIES = {
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


def build_dialogue_summary_prompt(
    scenario: Scenario, qa_history: List[dict], placements: List[List[str]]
) -> str:
    history = format_history(qa_history)
    placement_lines = "\n".join([f"- {obj}: {recep}" for obj, recep in placements])
    if not placement_lines:
        placement_lines = "None"
    return (
        STRICT_RULES
        + "* Output format: JSON that matches the provided schema.\n"
        "You are summarizing a robot's Q&A with a user about organizing objects.\n"
        "Summarize the user's preferences or placement rules in one concise sentence.\n"
        f"Room: {scenario.room}\n"
        f"Receptacles: {', '.join(scenario.receptacles)}\n"
        f"Q&A history:\n{history}\n"
        f"Known placements:\n{placement_lines}\n"
        "Return the summary in the JSON field only.\n"
    )


def query_placements(
    scenario: Scenario,
    objects: List[str],
    question_model: Optional[VLMAPI] = None,
    answer_model: Optional[VLMAPI] = None,
    question_strategy: str = "direct",
    question_builder: Optional[QuestionPromptBuilder] = None,
    max_questions: Optional[int] = None,
) -> Tuple[List[List[str]], List[dict], str]:
    question_model = question_model or VLMAPI("qwen3:32b")
    answer_model = answer_model or VLMAPI("qwen3:32b")
    question_builder = question_builder or QuestionPromptBuilder(question_strategy)
    question_builder.set_strategy(question_strategy)

    placements: List[List[str]] = []
    qa_history: List[dict] = []
    
    remaining_objects = list(objects)
    required_questions = resolve_max_questions(objects, max_questions)
    questions_asked = 0

    while questions_asked < required_questions:
        problem = build_problem_for_remaining(scenario, remaining_objects)
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
        questions_asked += 1
        
    # If we exited loop but still have objects (max turns reached), we might want to log that
    if remaining_objects:
        print(
            f"Warning: Failed to place {remaining_objects} after required questions "
            f"({questions_asked}/{required_questions})."
        )

    summary_prompt = build_dialogue_summary_prompt(scenario, qa_history, placements)
    summary_raw = _call_vlm(
        question_model,
        summary_prompt,
        "Output only JSON that matches the provided schema. Do NOT include reasoning or analysis.",
        temperature=0.0,
        format_schema=SummaryOutput.model_json_schema(),
    )
    structured_summary = parse_structured_output(summary_raw, SummaryOutput)
    summary_text = (
        structured_summary.summary if structured_summary else extract_answer(summary_raw)
    )

    return placements, qa_history, summary_text


def query_seen_placements(
    scenario: Scenario,
    question_model: Optional[VLMAPI] = None,
    answer_model: Optional[VLMAPI] = None,
    question_strategy: str = "direct",
    question_builder: Optional[QuestionPromptBuilder] = None,
    max_questions: Optional[int] = None,
) -> Tuple[List[List[str]], List[dict], str]:
    return query_placements(
        scenario,
        scenario.seen_objects,
        question_model=question_model,
        answer_model=answer_model,
        question_strategy=question_strategy,
        question_builder=question_builder,
        max_questions=max_questions,
    )


def evaluate_seen_placements(
    scenarios: List[Scenario],
    question_model: Optional[VLMAPI] = None,
    answer_model: Optional[VLMAPI] = None,
    scenario_limit: Optional[int] = None,
    question_strategy: str = "direct",
    question_builder: Optional[QuestionPromptBuilder] = None,
    max_questions: Optional[int] = None,
    log_path: Optional[str] = None,
    verbose: bool = True,
) -> float:
    question_model = question_model or VLMAPI("qwen3:32b")
    if answer_model is None:
        answer_model = question_model
    question_builder = question_builder or QuestionPromptBuilder(question_strategy)
    question_builder.set_strategy(question_strategy)
    resolved_log_path = resolve_log_path(log_path, question_strategy, max_questions)

    total_correct = 0
    total_items = 0

    for index, scenario in enumerate(scenarios[: scenario_limit or len(scenarios)]):
        placements, qa_history, summary_text = query_seen_placements(
            scenario,
            question_model=question_model,
            answer_model=answer_model,
            question_strategy=question_strategy,
            question_builder=question_builder,
            max_questions=max_questions,
        )
        corrects, accuracy = check_placements(placements, scenario.seen_placements)
        total_correct += sum(corrects)
        total_items += len(corrects)

        log_entry = asdict(scenario)
        placed_objects = {obj for obj, _ in placements}
        remaining_objects = [obj for obj in scenario.seen_objects if obj not in placed_objects]
        log_entry.update(
            {
                "scenario_index": index + 1,
                "strategy": question_strategy,
                "max_questions": resolve_max_questions(scenario.seen_objects, max_questions),
                "questions_asked": len(qa_history),
                "qa_history": qa_history,
                "predicted_placements": placements,
                "corrects": corrects,
                "accuracy": accuracy,
                "summary": summary_text,
                "remaining_objects": remaining_objects,
            }
        )
        append_test_log(resolved_log_path, log_entry)

        if verbose:
            print(f"\nScenario {index + 1}: {scenario.room}")
            print("Q&A history:")
            print(json.dumps(qa_history, ensure_ascii=True, indent=2))
            print("\nDialogue summary:")
            print(summary_text)
            print("\nPredicted placements:")
            print(json.dumps(placements, ensure_ascii=True, indent=2))
            print(f"\nScenario accuracy: {accuracy:.2f}")

    overall_accuracy = total_correct / total_items if total_items else 0.0
    if verbose:
        print(f"\nOverall accuracy: {overall_accuracy:.2f}")
        print(f"Log saved to: {resolved_log_path}")
    return overall_accuracy


def evaluate_accuracy_curve(
    scenarios: List[Scenario],
    strategies: List[str],
    max_questions_list: List[int],
    question_model: Optional[VLMAPI] = None,
    answer_model: Optional[VLMAPI] = None,
    log_dir: Optional[str] = None,
    plot_path: Optional[str] = None,
    parallel_strategies: bool = False,
    strategy_workers: Optional[int] = None,
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {}
    if parallel_strategies and len(strategies) > 1:
        max_workers = resolve_strategy_workers(strategy_workers, len(strategies))

        def run_strategy(strategy: str) -> Tuple[str, List[float]]:
            local_question_model = question_model or VLMAPI("qwen3:32b")
            local_answer_model = answer_model if answer_model is not None else local_question_model
            accuracies: List[float] = []
            question_builder = QuestionPromptBuilder(strategy)
            for max_questions in max_questions_list:
                log_path = None
                if log_dir:
                    log_path = os.path.join(
                        log_dir, f"test_log_{strategy}_q{max_questions}.json"
                    )
                accuracy = evaluate_seen_placements(
                    scenarios,
                    question_model=local_question_model,
                    answer_model=local_answer_model,
                    question_strategy=strategy,
                    question_builder=question_builder,
                    max_questions=max_questions,
                    log_path=log_path,
                    verbose=False,
                )
                accuracies.append(accuracy)
            return strategy, accuracies

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_strategy, s): s for s in strategies}
            for future in as_completed(futures):
                strategy, accuracies = future.result()
                results[strategy] = accuracies
    else:
        question_model = question_model or VLMAPI("qwen3:32b")
        if answer_model is None:
            answer_model = question_model
        for strategy in strategies:
            accuracies = []
            question_builder = QuestionPromptBuilder(strategy)
            for max_questions in max_questions_list:
                log_path = None
                if log_dir:
                    log_path = os.path.join(
                        log_dir, f"test_log_{strategy}_q{max_questions}.json"
                    )
                accuracy = evaluate_seen_placements(
                    scenarios,
                    question_model=question_model,
                    answer_model=answer_model,
                    question_strategy=strategy,
                    question_builder=question_builder,
                    max_questions=max_questions,
                    log_path=log_path,
                    verbose=False,
                )
                accuracies.append(accuracy)
            results[strategy] = accuracies

    if plot_path:
        plot_accuracy_curve(max_questions_list, results, plot_path)
        print(f"Accuracy curve saved to: {plot_path}")

    return results


def plot_accuracy_curve(
    max_questions_list: List[int],
    results: Dict[str, List[float]],
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot accuracy curves.") from exc

    plt.figure(figsize=(8, 5))
    for strategy, accuracies in results.items():
        plt.plot(max_questions_list, accuracies, marker="o", label=strategy)
    plt.xlabel("Max Question Turns")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Max Question Turns")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
        default="direct,user_preference,parallel",
        help="Comma-separated: direct,user_preference,parallel",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Required question turns per scenario (0 means auto).",
    )
    parser.add_argument(
        "--log_path",
        default="",
        help="Optional JSON log path (supports {strategy} and {max_questions}).",
    )
    parser.add_argument(
        "--curve_max_questions",
        default="",
        help="Comma-separated max question counts to plot accuracy curve.",
    )
    parser.add_argument(
        "--curve_output",
        default=os.path.join("summarization", "acc_curve.png"),
        help="Path to save accuracy curve plot.",
    )
    parser.add_argument(
        "--curve_log_dir",
        default=os.path.join("summarization", "test_logs"),
        help="Directory to save curve test logs.",
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
    max_questions = args.max_questions if args.max_questions > 0 else None
    overall_start = time.time()
    if args.curve_max_questions:
        curve_start = time.time()
        max_questions_list = [
            int(v) for v in args.curve_max_questions.split(",") if v.strip()
        ]
        print(
            f"Running accuracy curve for strategies={strategies} "
            f"max_questions={max_questions_list}"
        )
        curve_log_dir = args.curve_log_dir
        if curve_log_dir and not os.path.isabs(curve_log_dir):
            curve_log_dir = os.path.join(os.path.dirname(__file__), curve_log_dir)
        evaluate_accuracy_curve(
            scenarios,
            strategies=strategies,
            max_questions_list=max_questions_list,
            log_dir=curve_log_dir,
            plot_path=(
                args.curve_output
                if os.path.isabs(args.curve_output)
                else os.path.join(os.path.dirname(__file__), args.curve_output)
            ),
            parallel_strategies=args.parallel_strategies,
            strategy_workers=args.strategy_workers if args.strategy_workers > 0 else None,
        )
        curve_elapsed = time.time() - curve_start
        print(f"\nCurve evaluation time: {curve_elapsed:.2f}s")
        total_elapsed = time.time() - overall_start
        print(f"Total runtime: {total_elapsed:.2f}s")
        return

    if args.parallel_strategies and len(strategies) > 1:
        max_workers = resolve_strategy_workers(args.strategy_workers, len(strategies))
        print(f"\nRunning strategies in parallel (workers={max_workers})")

        def run_strategy(strategy: str) -> Tuple[str, float, str, float]:
            strategy_start = time.time()
            resolved_log_path = resolve_strategy_log_path(
                args.log_path or None,
                strategy,
                max_questions,
                parallel=True,
                strategy_count=len(strategies),
            )
            accuracy = evaluate_seen_placements(
                scenarios,
                question_model=None,
                answer_model=None,
                question_strategy=strategy,
                max_questions=max_questions,
                log_path=resolved_log_path,
                verbose=False,
            )
            elapsed = time.time() - strategy_start
            return strategy, accuracy, resolved_log_path, elapsed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_strategy, s): s for s in strategies}
            for future in as_completed(futures):
                strategy, accuracy, log_path, elapsed = future.result()
                print(f"\n=== Strategy: {strategy} ===")
                print(f"Overall accuracy: {accuracy:.2f}")
                print(f"Log saved to: {log_path}")
                print(f"Strategy time: {elapsed:.2f}s")
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
            evaluate_seen_placements(
                scenarios,
                question_model=question_model,
                answer_model=answer_model,
                question_strategy=strategy,
                max_questions=max_questions,
                log_path=log_path,
            )
            elapsed = time.time() - strategy_start
            print(f"Strategy time: {elapsed:.2f}s")

    total_elapsed = time.time() - overall_start
    print(f"\nTotal runtime: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
