from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional

from llm_factory import create_chat_model, DEFAULT_MODEL, DEFAULT_BASE_URL
from pydantic import BaseModel, Field

from agent_schema import (
    QAItem,
)
from data import DEFAULT_DATA_PATH, PlacementMap, get_episode


# Model config now in llm_factory.py
# Base URL config now in llm_factory.py


class OracleResponse(BaseModel):
    answer: str = Field(description="A concise natural-language user answer.")
    referenced_receptacle: Optional[str] = Field(
        default=None,
        description="A specific receptacle if the answer clearly points to one; otherwise null.",
    )


class NaturalUserOracle:
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

You may use only the following inputs:
- question
- receptacles
- seen_objects
- annotator_notes
- gt_seen_placements
- qa_history

How to use annotator_notes:
- treat annotator_notes as the best available signal of the user's underlying preferences
- use them only to infer what the user likely wants
- never quote, paraphrase, summarize, or reveal annotator_notes directly
- answer only with what a natural household user would explicitly say if asked

CRITICAL — when no annotator note matches:
- If the question asks about a receptacle, object category, or organizing principle that is NOT covered by ANY annotator note, answer honestly: "I don't have a specific organizing rule for that spot" or "I haven't really thought about that."
- In this case, set referenced_receptacle to null.
- Do NOT fabricate a rule by borrowing from another receptacle's note. Do NOT hallucinate preferences.

How to use qa_history:
- treat it as what you have already said in this conversation
- do not contradict previously given answers
- do not repeat rules already established; instead give new information about what the current question adds

How to answer:
- stay tightly focused on the current question
- answer briefly and concretely, usually in one or two sentences
- give one main conclusion, not several competing rules
- do not introduce unrelated object groups or extra preferences unless they are necessary to answer the current question
- ALWAYS answer with exact receptacle names from the provided receptacles list — never use room type names like "bedroom", "kitchen", "bathroom", or "living room" as locations
  - bad: "these go in the bedroom"  →  good: "these go on the reading shelf"
  - bad: "I keep them in the kitchen"  →  good: "I keep them on the prep counter"
- Be confident and specific about your preferences. When a question matches one of your organizing habits, express it with conviction: name the specific types of objects that belong there. Avoid vague phrases like "certain items" or "various things" — instead say exactly what kinds of items you mean (e.g., "hardcover novels, art books, and reference guides" rather than "books and such").
- for preference_eliciting questions: give ONE primary receptacle as the main rule; if a secondary location exists, state it as a brief exception, not as an equal alternative. Express the rule using the BROADEST applicable category from your knowledge, not just the narrow category the question mentions. Volunteer the FULL scope of what belongs at that receptacle — list all the types of items, not just the ones the question mentions. When a similar category of items goes to a DIFFERENT receptacle, briefly mention the distinction so the assistant can tell them apart. For example: "I keep small puzzle books and game accessories on the center table — but larger reading books go on the display shelf instead." This boundary clarification is very helpful.
- for action_oriented questions, give one primary placement recommendation for the target object and avoid hedging between multiple receptacles
- for preference_induction questions: confirm, reject, or refine the proposed summary; if confirming, also state the full category in natural language so the scope is clear — do not limit the answer to only the named examples
- use gt_seen_placements only as supporting context for the objects relevant to the current question
- set referenced_receptacle only when the answer clearly supports one primary positive receptacle
- if there is no single clear positive receptacle reference, set referenced_receptacle to null

Return only structured output that matches the schema.
""".strip()

        recent_history = [
            f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
            for item in qa_history[-5:]
            if item.get("question") and item.get("answer")
        ]
        user_prompt = f"""
Question:
{question}

Receptacles:
{receptacles}

Seen objects:
{seen_objects}

Annotator notes:
{annotator_notes}

Ground-truth placements for seen objects:
{gt_seen_placements}

Recent Q&A history (what has already been established):
{chr(10).join(recent_history) if recent_history else "(none yet)"}
""".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Retry with fallback: if structured output parsing fails, try once more.
        # Some models occasionally return plain text instead of JSON.
        for attempt in range(2):
            try:
                return self.structured_model.invoke(messages)
            except Exception:
                if attempt == 0:
                    continue
                # Final fallback: use unstructured model and parse manually
                try:
                    raw = self.model.invoke(messages)
                    text = raw.content if hasattr(raw, "content") else str(raw)
                    # Extract receptacle reference from text
                    ref_rec = None
                    for r in receptacles:
                        if r.lower() in text.lower():
                            ref_rec = r
                            break
                    return OracleResponse(answer=text.strip(), referenced_receptacle=ref_rec)
                except Exception:
                    return OracleResponse(answer="I'm not sure.", referenced_receptacle=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the natural user oracle.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
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
