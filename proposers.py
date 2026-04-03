from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any, List, Literal, Optional, TypedDict

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent_schema import (
    AgentState,
)
from data import get_episode
from state_init import build_initial_state
from state_update import StateUpdate


QUESTION_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "scenarios_aug_tiny.json"


class PreferenceQuestionIntent(TypedDict, total=False):
    question_pattern: Literal["preference_eliciting", "preference_induction"]
    hypothesis: str
    covered_objects: List[str]
    receptacle: Optional[str]
    priority: float
    question: str


class PreferenceQuestionIntentModel(BaseModel):
    question_pattern: Literal["preference_eliciting", "preference_induction"]
    hypothesis: str = Field(description="A concise preference hypothesis worth asking the user to clarify.")
    covered_objects: List[str] = Field(
        description="Seen objects likely affected by this preference intent."
    )
    receptacle: Optional[str] = Field(
        default=None,
        description="An optional exact receptacle name if the model has a plausible likely placement; otherwise null.",
    )
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A short natural user-facing question for this preference intent.")


class PreferenceQuestionIntentBatch(BaseModel):
    intents: List[PreferenceQuestionIntentModel] = Field(
        description="A small conservative list of preference question intents for the next turn."
    )


class ElicitingQuestionIntentModel(BaseModel):
    hypothesis: str = Field(description="A concise preference hypothesis worth asking the user to clarify.")
    covered_objects: List[str] = Field(
        description="Seen objects likely affected by this preference intent."
    )
    receptacle: Optional[str] = Field(
        default=None,
        description="An optional exact receptacle name if the model has a plausible likely placement; otherwise null.",
    )
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A short natural user-facing question for this preference intent.")


class BuiltPreferenceCandidateModel(BaseModel):
    hypothesis: str = Field(
        description="A short, usable preference hypothesis phrase that clearly covers the listed covered_objects."
    )
    covered_objects: List[str] = Field(
        default_factory=list,
        description="One to three exact seen object names that are clear anchor examples of the hypothesis.",
    )


class BuiltPreferenceCandidateBatch(BaseModel):
    candidates: List[BuiltPreferenceCandidateModel] = Field(
        default_factory=list,
        description="A small list of structured preference candidates for future preference-eliciting questions.",
    )


class ActionIntent(BaseModel):
    question_pattern: Literal["action_oriented"] = "action_oriented"
    action_mode: Literal["direct_grounding", "boundary_probe"] = "direct_grounding"
    object_name: str = Field(description="One exact unresolved seen object name to ask about next.")
    priority: float = Field(description="Importance score from 0.0 to 1.0.")
    question: str = Field(description="A direct action-oriented question for this object.")


# =========================================================
# Shared helpers
# =========================================================

def _clip_priority(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# =========================================================
# Preference-eliciting proposer
# =========================================================

class PreferenceElicitingProposer:
    """
    Scene-derived proposer for Preference-eliciting questions.

    Input:
    - room
    - receptacles
    - seen_objects
    Output:
    - a small list of preference question intents
    - each intent already includes a natural question
    """

    def __init__(
        self,
        model: str = QUESTION_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.model: Any = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
        )
        self.structured_model = self.model.with_structured_output(
            ElicitingQuestionIntentModel
        )
        self.candidate_model = self.model.with_structured_output(
            BuiltPreferenceCandidateBatch
        )

    def _build_preference_candidates(
        self,
        *,
        state: AgentState,
        max_candidates: int = 1,
    ) -> List[BuiltPreferenceCandidateModel]:
        confirmed_hypotheses = [
            item.get("hypothesis", "").strip()
            for item in state["confirmed_preferences"]
            if item.get("hypothesis", "").strip()
        ]
        negative_hypotheses = [
            item.get("hypothesis", "").strip()
            for item in state["negative_preferences"]
            if item.get("hypothesis", "").strip()
        ]

        # Objects already addressed: covered by a confirmed preference rule OR in confirmed_actions
        covered_by_preferences: set[str] = set()
        for item in state["confirmed_preferences"]:
            covered_by_preferences.update(item.get("covered_objects", []))
        confirmed_action_objects = {item["object_name"] for item in state["confirmed_actions"]}
        already_handled = covered_by_preferences | confirmed_action_objects

        # Only ask about objects not yet addressed by any rule or direct action
        genuinely_unresolved = [
            obj for obj in state["unresolved_objects"]
            if obj not in already_handled
        ]
        if not genuinely_unresolved:
            return []

        system_prompt = f"""
Generate preference candidates for future preference-eliciting questions in a household tidying task.

Return at most {max_candidates} structured candidates based on the unresolved objects.

Field requirements:
- hypothesis: a SHORT category label (2-5 words). It names a grouping of objects, NOT a placement.
- covered_objects: several exact unresolved object names that are strong examples of the same hypothesis.

CRITICAL hypothesis rules:
- hypothesis must be a category label only — do NOT include receptacle names, verbs, or placement instructions
- hypothesis must NOT contain words like "should", "belong", "go", "place", "put", "stored", or any room/receptacle name
- if you know where items go, keep that knowledge to yourself — only name the category

Style examples:
- good: "bedside-use items"
- good: "powered kitchen tools"
- good: "reading materials"
- good: "electronic accessories"
- good: "toy and game storage"
- bad: "electronics should be placed in the media console" — contains receptacle name and verb
- bad: "books go on the display shelf" — contains receptacle and verb
- bad: "storage items for the ottoman" — contains receptacle name
- bad: "Should electronics be grouped together?" — question form
- bad: "Electrical appliances vs non-electrical items" — comparison
- bad: "Use frequency" — too vague
""".strip()

        user_prompt = f"""
Room:
{state["room"]}

Receptacles:
{state["receptacles"]}

Seen objects:
{state["seen_objects"]}

Unresolved objects (still needing a preference question — do NOT generate hypotheses for objects already addressed):
{genuinely_unresolved}

Confirmed actions:
{state["confirmed_actions"]}

Confirmed preference hypotheses (already asked — do not repeat or rephrase these):
{confirmed_hypotheses}

Negative preference hypotheses (rejected — do not repeat):
{negative_hypotheses}

Return only structured candidates.
Focus on unresolved objects and infer short, usable preference hypotheses.
The hypothesis may be based on grouping, function, use context, or another realistic organizing dimension, but it should still read like something a user could answer directly.
""".strip()

        try:
            result = self.candidate_model.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            raw_candidates = result.candidates
        except Exception:
            raw_candidates = []

        rejected = {
            item.get("hypothesis", "").strip().lower()
            for item in state["negative_preferences"]
            if item.get("hypothesis", "").strip()
        }
        confirmed = {
            item.get("hypothesis", "").strip().lower()
            for item in state["confirmed_preferences"]
            if item.get("hypothesis", "").strip()
        }
        genuinely_unresolved_set = set(genuinely_unresolved)
        seen_objects_norm = {item.strip().lower() for item in state["seen_objects"] if item.strip()}
        receptacles_norm = {item.strip().lower() for item in state["receptacles"] if item.strip()}
        deduped: List[BuiltPreferenceCandidateModel] = []
        seen = set()
        for item in raw_candidates:
            hypothesis = item.hypothesis.strip()
            normalized = " ".join(hypothesis.lower().strip().split())
            if not normalized:
                continue
            if normalized in seen or normalized in rejected:
                continue
            # Substring dedup: skip if candidate phrase is contained in any confirmed/rejected hypothesis
            # (handles mismatch between short candidate label and stored long-form hypothesis text)
            if any(normalized in c for c in confirmed) or any(normalized in r for r in rejected):
                continue
            if normalized in seen_objects_norm or normalized in receptacles_norm:
                continue
            if any(obj in normalized for obj in seen_objects_norm):
                continue
            if any(rec in normalized for rec in receptacles_norm):
                continue
            covered_objects = [
                obj for obj in _dedupe_keep_order(list(item.covered_objects))
                if obj in genuinely_unresolved_set
            ][:3]
            if not covered_objects:
                continue
            # Object-overlap dedup: skip only if all covered_objects already appear in an accepted candidate
            # (prevents exact-duplicate candidates; allows partial-overlap candidates through)
            covered_set = set(covered_objects)
            if deduped and covered_set.issubset(
                {obj for existing in deduped for obj in existing.covered_objects}
            ):
                continue
            deduped.append(
                BuiltPreferenceCandidateModel(
                    hypothesis=hypothesis,
                    covered_objects=covered_objects,
                )
            )
            seen.add(normalized)
            if len(deduped) >= max_candidates:
                break

        return deduped

    def _propose_from_candidates(
        self,
        *,
        state: AgentState,
        candidates: List[BuiltPreferenceCandidateModel],
        guidance: str = "",
    ) -> Optional[PreferenceQuestionIntent]:
        if not candidates:
            return None

        system_prompt = """
Choose preference-eliciting questions for a household rearrangement task.

Rules:
- Choose only from the given preference candidates.
- Do not invent a new grouping.
- Keep the hypothesis exactly as given.
- Keep covered_objects within the candidate's covered_objects.
- Prefer candidates that can clarify more than one unresolved object.
- Return exactly one best preference-eliciting intent.

Question goal:
- The question must reveal the user's organizing HABIT or PRINCIPLE for that category — not just ask "where does it go".
- The question MUST reference the hypothesis category name. Covered_objects may appear as clarifying examples only.
- Ask how the user tends to organize, whether items stay in one spot, or why they assign things to a particular area.
- Do NOT ask "where should X be placed?" — that is identical to an action-oriented question.
- Do NOT ask whether items should be grouped together.

Style examples:
- good: "How do you usually organize electronic accessories — do they tend to stay in one spot?"
- good: "Is there a specific area you always assign to reading materials like books or magazines?"
- good: "Do personal care items generally stay together for you, or spread around based on use?"
- good: "Should powered kitchen tools be kept in one place, or do some live near where they're used?"
- good (with receptacle hint): "Do media devices usually end up in the media console for you?"
- bad: "Where should media devices usually be placed?" — asks WHERE not HOW/WHY
- bad: "Where do personal care items go?" — action-oriented phrasing
- bad: "Would you like to group personal care items together?" — grouping question
""".strip()

        user_prompt = f"""
Preference candidates:
{candidates}

Unresolved objects:
{state["unresolved_objects"]}

Guidance:
{guidance}

Return exactly one intent.

Fields:
- hypothesis = exactly one hypothesis from the given preference candidates
- covered_objects = exact objects chosen only from that candidate's covered_objects
- receptacle = optional exact receptacle name from the room if you have a plausible likely placement, otherwise null
- priority = 0.0 to 1.0
- question = one short question that elicits the user's organizing habit or principle for the hypothesis category
- if receptacle is null, ask an open principle question ("How do you usually organize [category]?" / "Is there a spot you always assign to [category]?")
- if receptacle is non-null, ask a soft confirmation ("Do [category] usually end up in [receptacle] for you?")
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_preference_eliciting_intent(
            intent=result,
            state=state,
            candidates=candidates,
        )

    def propose(
        self,
        *,
        state: AgentState,
        guidance: str = "",
        max_candidates: int = 5,
    ) -> Optional[PreferenceQuestionIntent]:
        candidates = self._build_preference_candidates(
            state=state,
            max_candidates=max_candidates,
        )
        return self._propose_from_candidates(
            state=state,
            candidates=candidates,
            guidance=guidance,
        )


def _normalize_preference_eliciting_intent(
    *,
    intent: ElicitingQuestionIntentModel,
    state: AgentState,
    candidates: List[BuiltPreferenceCandidateModel],
 ) -> Optional[PreferenceQuestionIntent]:
    allowed_objects = set(state["seen_objects"])
    open_hypotheses = {
        item.hypothesis.strip().lower()
        for item in candidates
        if item.hypothesis.strip()
    }
    allowed_candidate_objects = {
        item.hypothesis.strip().lower(): set(item.covered_objects)
        for item in candidates
        if item.hypothesis.strip()
    }
    if not open_hypotheses:
        return None

    hypothesis = intent.hypothesis.strip()
    question = intent.question.strip()
    if not hypothesis or not question:
        return None
    if open_hypotheses and hypothesis.lower() not in open_hypotheses:
        return None

    covered_objects = [
        obj for obj in _dedupe_keep_order(list(intent.covered_objects))
        if obj in allowed_objects
    ]
    candidate_objects = allowed_candidate_objects.get(hypothesis.lower(), set())
    if candidate_objects:
        covered_objects = [
            obj for obj in covered_objects
            if obj in candidate_objects
        ]
    if not covered_objects:
        return None

    return PreferenceQuestionIntent(
        question_pattern="preference_eliciting",
        hypothesis=hypothesis,
        covered_objects=covered_objects,
        receptacle=intent.receptacle.strip() if intent.receptacle and intent.receptacle.strip() in state["receptacles"] else None,
        priority=_clip_priority(intent.priority),
        question=question,
    )


# =========================================================
# Action-oriented proposer
# =========================================================

class ActionProposer:
    """
    Proposer for Action-oriented questions.

    This proposer covers two internal modes:
    - direct_grounding
    - boundary_probe

    Both remain within the same empirical pattern:
    - question_pattern = "action_oriented"
    """

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
            reasoning=False,
        )
        self.structured_model = self.model.with_structured_output(ActionIntent)

    def propose(
        self,
        *,
        state: AgentState,
        guidance: str = "",
    ) -> Optional[ActionIntent]:
        if not state["unresolved_objects"]:
            return None

        system_prompt = """
You are a proposer for Action-oriented questions in a household rearrangement task.

Your job:
Choose ONE unresolved seen object that is the best next target for an Action-oriented question.

Object selection strategy — apply in order:
1. boundary_probe candidate: if a confirmed_preference exists, pick an unresolved object that is semantically
   similar to its covered objects. Asking about it tests whether the preference extends further and builds
   richer analogy evidence for unseen objects.
2. category anchor: if no confirmed_preference exists yet, pick an object that is representative of a
   common household category (electronics, books, cleaning supplies, etc.) so the answer can serve as
   an analogy anchor for other similar unresolved objects.
3. fallback: any remaining unresolved object.

Action modes:
- direct_grounding: ask where the object should go (use when no related confirmed preference exists)
- boundary_probe: ask whether the object follows an existing confirmed preference rule (use when a
  plausible preference exists whose scope is uncertain)

Question rules:
- must mention the chosen object by name
- for direct_grounding: "Where should the [object] go?" or equivalent
- for boundary_probe: "Does the [object] also belong in [receptacle from confirmed preference]?"
- one sentence, no hedging, no unrelated objects
""".strip()

        recent_qa_history = state["qa_history"][-3:]

        user_prompt = f"""
Unresolved seen objects:
{state["unresolved_objects"]}

Confirmed actions:
{state["confirmed_actions"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Guidance:
{guidance}

Recent QA history:
{recent_qa_history}

Return exactly one ActionIntent:
- question_pattern = "action_oriented"
- action_mode = "direct_grounding" or "boundary_probe"
- object_name = one exact unresolved seen object
- priority = 0.0 to 1.0
- question = one concise natural Action-oriented question about that exact object's placement or boundary case
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_action_intent(
            intent=result,
            state=state,
        )


def _normalize_action_intent(
    *,
    intent: ActionIntent,
    state: AgentState,
) -> Optional[ActionIntent]:
    object_name = intent.object_name.strip()
    question = intent.question.strip()

    if not object_name or not question:
        return None
    if object_name not in state["unresolved_objects"]:
        return None
    if object_name in {item["object_name"] for item in state["confirmed_actions"]}:
        return None

    action_mode = intent.action_mode
    if action_mode not in ("direct_grounding", "boundary_probe"):
        action_mode = "direct_grounding"

    return ActionIntent(
        question_pattern="action_oriented",
        action_mode=action_mode,
        object_name=object_name,
        priority=_clip_priority(intent.priority),
        question=question,
    )


# =========================================================
# Preference-induction proposer
# =========================================================

class PreferenceInductionProposer:
    """
    Proposer for Preference-summary questions.

    These questions summarize a candidate rule inferred from existing evidence
    and ask the user to confirm or refine it.
    """

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
            reasoning=False,
        )
        self.structured_model = self.model.with_structured_output(
            PreferenceQuestionIntentBatch
        )

    def propose(
        self,
        *,
        state: AgentState,
        max_intents: int = 3,
        guidance: str = "",
    ) -> List[PreferenceQuestionIntent]:

        system_prompt = f"""
You are a proposer for preference-summary questions in a household rearrangement task.

Your job:
Propose a small number of high-value summary questions that are worth spending budget on.

Choose summaries whose confirmation is most likely to:
- explain multiple confirmed_actions or unresolved objects
- compress existing object-level evidence into a useful placement rule
- reduce uncertainty not already resolved by confirmed preferences

Avoid summaries that:
- restate an already confirmed preference
- are too weak or too narrow to affect future placement decisions
- only describe one object unless no broader summary remains

Rules:
- This pattern is always "preference_induction".
- Do not output action-oriented questions.
- Do not output preference-eliciting questions.
- Return at most {max_intents} intents.
- Use only exact seen object names in covered_objects.
- Use the guidance as a soft instruction about what kind of summary is most useful to confirm next.
""".strip()

        user_prompt = f"""
Confirmed actions:
{state["confirmed_actions"]}

Seen objects:
{state["seen_objects"]}

Unresolved objects:
{state["unresolved_objects"]}

Confirmed preferences:
{state["confirmed_preferences"]}

Rejected hypotheses:
{state["negative_preferences"]}

Guidance:
{guidance}

Choose the most useful summary question, not just a plausible one.
Use confirmed_actions as the main evidence source, and prefer a summary whose confirmation would improve future placement decisions.

Each intent must include:
- question_pattern = "preference_induction"
- hypothesis = a concise summary rule to confirm
- covered_objects = exact seen objects plausibly covered by the summary
- receptacle = optional exact receptacle name if there is a clear likely placement to confirm, otherwise null
- priority = 0.0 to 1.0
""".strip()

        result = self.structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _normalize_preference_induction_intents(
            intents=result.intents,
            state=state,
            max_intents=max_intents,
        )


def _normalize_preference_induction_intents(
    *,
    intents: List[PreferenceQuestionIntentModel],
    state: AgentState,
    max_intents: int,
) -> List[PreferenceQuestionIntent]:
    allowed_objects = set(state["seen_objects"])
    existing_hypotheses = {
        item.get("hypothesis", "").strip().lower()
        for item in state["confirmed_preferences"] + state["negative_preferences"]
        if item.get("hypothesis", "").strip()
    }
    negative_preferences = {
        item.get("hypothesis", "").strip().lower()
        for item in state["negative_preferences"]
        if item.get("hypothesis", "").strip()
    }

    normalized: List[PreferenceQuestionIntent] = []
    seen_signatures = set()

    for item in intents:
        hypothesis = item.hypothesis.strip()
        question = item.question.strip()
        if not hypothesis or not question:
            continue
        if hypothesis.lower() in existing_hypotheses:
            continue
        if hypothesis.lower() in negative_preferences:
            continue

        covered_objects = [
            obj for obj in _dedupe_keep_order(list(item.covered_objects))
            if obj in allowed_objects
        ]
        if len(covered_objects) < 2:
            continue

        signature = (
            hypothesis.lower(),
            tuple(sorted(covered_objects)),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        normalized.append(
            PreferenceQuestionIntent(
                question_pattern="preference_induction",
                hypothesis=hypothesis,
                covered_objects=covered_objects,
                receptacle=item.receptacle.strip() if item.receptacle and item.receptacle.strip() in state["receptacles"] else None,
                priority=_clip_priority(item.priority),
                question=question,
            )
        )

    normalized.sort(key=lambda x: float(x.get("priority", 0.0)), reverse=True)
    return normalized[:max_intents]


# =========================================================
# Convenience wrappers
# =========================================================

def propose_preference_eliciting_intent(
    *,
    state: AgentState,
    proposer: PreferenceElicitingProposer,
    guidance: str = "",
    max_candidates: int = 5,
) -> Optional[PreferenceQuestionIntent]:
    return proposer.propose(
        state=state,
        guidance=guidance,
        max_candidates=max_candidates,
    )


def propose_action_intent(
    *,
    state: AgentState,
    proposer: ActionProposer,
) -> Optional[ActionIntent]:
    return proposer.propose(state=state)


def propose_preference_induction_intents(
    *,
    state: AgentState,
    proposer: PreferenceInductionProposer,
    max_intents: int = 3,
) -> List[PreferenceQuestionIntent]:
    return proposer.propose(state=state, max_intents=max_intents)


# =========================================================
# CLI / smoke test
# =========================================================

def main() -> None:
    from state_update import StateUpdate

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to dataset JSON file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Episode index.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="parallel_exploration",
        choices=["direct", "preference_first", "parallel_exploration"],
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["eliciting", "action", "summary", "all"],
        help="Which proposer(s) to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=QUESTION_MODEL,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=OLLAMA_BASE_URL,
    )
    args = parser.parse_args()

    episode = get_episode(args.data, index=args.index)
    state = build_initial_state(
        episode=episode,
        strategy=args.strategy,  # type: ignore[arg-type]
        budget_total=args.budget,
    )
    print("=== Current State ===")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print()

    eliciting_proposer = PreferenceElicitingProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )
    action_proposer = ActionProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )
    induction_proposer = PreferenceInductionProposer(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )

    if args.mode in ("eliciting", "all"):
        print("=== Preference-eliciting proposer ===")
        intent = propose_preference_eliciting_intent(
            state=state,
            proposer=eliciting_proposer,
        )
        print(intent)
        print()

    if args.mode in ("action", "all"):
        print("=== Action-oriented proposer ===")
        intent = propose_action_intent(
            state=state,
            proposer=action_proposer,
        )
        print(intent.model_dump() if intent is not None else None)
        print()

    if args.mode in ("summary", "all"):
        print("=== Preference-induction proposer ===")
        # Mock a little evidence so summary proposer has something to infer from
        state["confirmed_actions"] = [
            {"object_name": state["seen_objects"][0], "receptacle": state["receptacles"][0]},
            {"object_name": state["seen_objects"][1], "receptacle": state["receptacles"][0]},
        ]
        print("=== Current State For Preference-summary ===")
        print(json.dumps(state, indent=2, ensure_ascii=False))
        print()
        intents = propose_preference_induction_intents(
            state=state,
            proposer=induction_proposer,
            max_intents=3,
        )
        for intent in intents:
            print(intent)
        print()


if __name__ == "__main__":
    main()
