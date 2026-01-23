#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle answerer: goal-visible and QA-history-aware.
"""

import json
import re
from typing import Any, Dict, List, Optional

from ollama_call import VLMAPI


def parse_json_maybe(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def format_objects(objects: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in objects.items()])


def format_receptacles(receptacles: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in receptacles.items()])


def format_facts(facts: List[str]) -> str:
    return "\n".join([f"- {x}" for x in facts])


class OracleAnswerer:
    def __init__(self, model: str, options: Optional[Dict[str, Any]] = None) -> None:
        self.api = VLMAPI(model)
        self.options = options or {}
        self.answer_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

    def build_prompt(
        self,
        sample: Dict[str, Any],
        question: str,
    ) -> Dict[str, str]:
        objects = format_objects(sample.get("objects", {}))
        receptacles = format_receptacles(sample.get("receptacles", {}))
        goal = format_facts(sample.get("goal", []))

        systext = (
            "You are the user being asked about a household task.\n"
            "You can read only: task, objects, receptacles, and goal.\n"
            "Answer based ONLY on the goal facts (ignore any other info).\n"
            "Keep it brief, specific, and consistent with the goal.\n"
            "Explicitly state the placement location for the object(s) mentioned in the question.\n"
            "Output must be strict JSON with exactly one key: \"answer\".\n"
        )
        usertext = (
            "Context: The robot has limited chances to learn your preferences.\n\n"
            f"Task: {sample.get('task', 'unknown')}\n\n"
            f"Objects:\n{objects}\n\n"
            f"Receptacles:\n{receptacles}\n\n"
            f"Goal facts:\n{goal}\n\n"
            f"Question: {question}\n"
        )
        return {"systext": systext, "usertext": usertext}

    def answer(
        self,
        sample: Dict[str, Any],
        question: str,
    ) -> str:
        prompt = self.build_prompt(sample, question)
        text = self.api.vlm_request_with_format(
            prompt["systext"],
            prompt["usertext"],
            format_schema=self.answer_schema,
            options=self.options,
        )
        parsed = parse_json_maybe(text)
        answer = parsed.get("answer", "").strip()
        return answer or "Please place items according to the goal locations."
