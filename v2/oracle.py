from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

try:
    from v2.agent_schema import (
        QAItem,
    )
    from v2.data import DEFAULT_DATA_PATH, PlacementMap, get_episode
except ModuleNotFoundError:
    from agent_schema import (
        QAItem,
    )
    from data import DEFAULT_DATA_PATH, PlacementMap, get_episode


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class OracleResponse(BaseModel):
    answer: str = Field(description="A concise natural-language user answer.")
    referenced_receptacle: Optional[str] = Field(
        default=None,
        description="A specific receptacle if the answer clearly points to one; otherwise null.",
    )


class NaturalUserOracle:
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
        self.structured_model = self.model.with_structured_output(OracleResponse)

    def answer(
        self,
        *,
        question: str,
        room: str,
        receptacles: List[str],
        seen_objects: List[str],
        annotator_notes: List[str],
        gt_seen_placements: PlacementMap,
        qa_history: List[QAItem],
    ) -> OracleResponse:
        system_prompt = """
You are simulating a natural household user in a rearrangement task.

You may use ONLY the following human-visible information:
- question
- room
- receptacles
- seen_objects
- annotator_notes
- gt_seen_placements
- qa_history

Do not use any agent-internal metadata.
Do not mention hidden reasoning.
Answer naturally and briefly.
If the answer clearly names one receptacle, set referenced_receptacle to that exact receptacle.
Otherwise set referenced_receptacle to null.
""".strip()

        user_prompt = f"""
Question:
{question}

Room:
{room}

Receptacles:
{receptacles}

Seen objects:
{seen_objects}

Annotator notes:
{annotator_notes}

Ground-truth placements for seen objects:
{gt_seen_placements}

Previous QA history:
{qa_history}
""".strip()

        return self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the natural user oracle.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default=QUESTION_MODEL)
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL)
    args = parser.parse_args()

    episode = get_episode(Path(args.data), args.index)
    oracle = NaturalUserOracle(model=args.model, base_url=args.base_url, temperature=0.0)

    action_question = f"Where should I place the {episode.seen_objects[0]}?"
    action_response = oracle.answer(
        question=action_question,
        room=episode.room,
        receptacles=episode.receptacles,
        seen_objects=episode.seen_objects,
        annotator_notes=episode.annotator_notes,
        gt_seen_placements=episode.seen_placements,
        qa_history=[],
    )

    eliciting_question = "Do you prefer to keep fragile drinkware together or prioritize easy access?"
    eliciting_response = oracle.answer(
        question=eliciting_question,
        room=episode.room,
        receptacles=episode.receptacles,
        seen_objects=episode.seen_objects,
        annotator_notes=episode.annotator_notes,
        gt_seen_placements=episode.seen_placements,
        qa_history=[],
    )

    summary_question = "It sounds like you want drinkware grouped in the upper cabinet. Is that right?"
    summary_response = oracle.answer(
        question=summary_question,
        room=episode.room,
        receptacles=episode.receptacles,
        seen_objects=episode.seen_objects,
        annotator_notes=episode.annotator_notes,
        gt_seen_placements=episode.seen_placements,
        qa_history=[],
    )

    print("=== Oracle Smoke ===")
    print(
        json.dumps(
            {
                "action": action_response.model_dump(),
                "eliciting": eliciting_response.model_dump(),
                "summary": summary_response.model_dump(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
