from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any, List, Literal, Optional, TypedDict

from llm_factory import create_chat_model, DEFAULT_MODEL, DEFAULT_BASE_URL
from pydantic import BaseModel, Field

from agent_schema import (
    AgentState,
)
from data import get_episode
from state_init import build_initial_state
from state_update import StateUpdate


# Model config now in llm_factory.py
# Base URL config now in llm_factory.py
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
        description="ALL exact unresolved object names that belong to this hypothesis category. Include every matching object.",
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
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.model: Any = create_chat_model(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
            timeout=120,
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
- hypothesis: a descriptive category label (3-8 words) that names a grouping of objects by shared attributes. Include distinguishing attributes such as material (glass, ceramic, wooden), size (small, handheld, large), power source (plug-in, battery-powered, cordless), or usage context (bedside, cooking, reading). Do NOT include receptacle names, verbs, or placement instructions.
- covered_objects: ALL exact unresolved object names that fit this hypothesis. Include every matching object, not just a few examples.

CRITICAL hypothesis rules:
- hypothesis must describe a category using object attributes — NOT just a one-word type label
- hypothesis must NOT contain receptacle names, verbs like "should", "belong", "go", "place", "put", "stored"
- if you know where items go, keep that knowledge to yourself — only name the category with its attributes

Style examples:
- good: "fragile glass and ceramic drinkware"
- good: "small battery-powered handheld devices"
- good: "plug-in bedside electronics"
- good: "soft comfort textiles and throws"
- good: "hardcover reading and reference books"
- good: "small handheld prep tools"
- bad: "electronics" — too vague, no distinguishing attributes
- bad: "lighting tools" — too generic, misses power-source or size
- bad: "electronics should be placed in the media console" — contains receptacle name and verb
- bad: "books go on the display shelf" — contains receptacle and verb
- bad: "storage items for the ottoman" — contains receptacle name
""".strip()

        covered_receptacles = sorted({
            cp.get("receptacle") for cp in state["confirmed_preferences"] if cp.get("receptacle")
        })
        uncovered_receptacles = sorted({r for r in state["receptacles"] if r not in set(covered_receptacles)})

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

Receptacles not yet covered by any confirmed preference: {uncovered_receptacles}
Hint: prefer hypotheses whose covered_objects are likely to belong to one of the uncovered receptacles above.

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
            ]
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

        # Identify which candidates are receptacle-centric (empty covered_objects)
        has_receptacle_centric = any(len(c.covered_objects) == 0 for c in candidates)

        receptacle_centric_block = ""
        if has_receptacle_centric:
            receptacle_centric_block = """
IMPORTANT — Receptacle-centric candidates:
- Some candidates have empty covered_objects. These target receptacles not yet covered by any rule.
- Prefer receptacle-centric candidates when available — they discover entirely unknown organizing rules.
- Among multiple receptacle-centric candidates, pick the one most likely to cover the MOST unresolved objects.
  Think about which receptacle would typically hold the largest variety of items in this room.
- For receptacle-centric candidates, the question MUST be phrased as:
  "What kinds of items do you typically keep in/on the [receptacle]?"
- Set covered_objects to [] (empty) for receptacle-centric candidates.
- Set receptacle to the exact receptacle name from the hypothesis.
"""

        system_prompt = f"""
Choose preference-eliciting questions for a household rearrangement task.

Rules:
- Choose only from the given preference candidates.
- Do not invent a new grouping.
- Keep the hypothesis exactly as given.
- Keep covered_objects within the candidate's covered_objects.
- Prefer candidates that can clarify more than one unresolved object.
- Return exactly one best preference-eliciting intent.
{receptacle_centric_block}
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
- good (receptacle-centric): "What kinds of items do you typically keep in the storage ottoman?"
- good (receptacle-centric): "What do you usually store on the display shelf?"
- bad: "Where should media devices usually be placed?" — asks WHERE not HOW/WHY
- bad: "Where do personal care items go?" — action-oriented phrasing
- bad: "Would you like to group personal care items together?" — grouping question
""".strip()

        covered_receptacles_propose = sorted({
            cp.get("receptacle") for cp in state["confirmed_preferences"] if cp.get("receptacle")
        })
        uncovered_receptacles_propose = sorted({r for r in state["receptacles"] if r not in set(covered_receptacles_propose)})

        user_prompt = f"""
Preference candidates:
{candidates}

Unresolved objects:
{state["unresolved_objects"]}

Receptacles not yet covered by any confirmed preference: {uncovered_receptacles_propose}

Guidance:
{guidance}

Return exactly one intent.

Fields:
- hypothesis = exactly one hypothesis from the given preference candidates
- covered_objects = exact objects chosen only from that candidate's covered_objects (use [] for receptacle-centric candidates with empty covered_objects)
- receptacle = best-guess exact receptacle from the receptacles list based on common household knowledge.
  STRONGLY PREFER providing a specific receptacle so the question can be a soft confirmation.
  For receptacle-centric candidates, set this to the exact receptacle name from the hypothesis.
  Only leave null if you genuinely cannot guess any plausible placement.
- priority = 0.0 to 1.0
- question = one short natural question — refer to the style examples above for phrasing options
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

    def _build_receptacle_centric_candidates(
        self,
        *,
        state: AgentState,
        max_candidates: int = 2,
    ) -> List[BuiltPreferenceCandidateModel]:
        """Generate candidates for gap receptacles (no seen objects, no preferences)."""
        covered_recs = {cp.get("receptacle") for cp in state["confirmed_preferences"] if cp.get("receptacle")}
        covered_recs |= {ca.get("receptacle") for ca in state["confirmed_actions"] if ca.get("receptacle")}
        # Also count receptacles already asked about in qa_history
        for qa in state["qa_history"]:
            if qa.get("question_pattern") == "preference_eliciting":
                target = qa.get("target", "")
                for r in state["receptacles"]:
                    if r.lower() in target.lower():
                        covered_recs.add(r)
        uncovered_recs = [r for r in state["receptacles"] if r not in covered_recs]
        if not uncovered_recs:
            return []

        candidates = []
        for r in uncovered_recs:
            candidates.append(
                BuiltPreferenceCandidateModel(
                    hypothesis=f"items typically kept in the {r}",
                    covered_objects=[],  # gap receptacle: no seen objects to list
                )
            )
        return candidates

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

        # Add receptacle-centric candidates for gap receptacles
        rc_candidates = self._build_receptacle_centric_candidates(
            state=state,
            max_candidates=2,
        )

        # When gap receptacles exist, prioritize them by putting them first
        if rc_candidates:
            candidates = rc_candidates + candidates

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
    # Track which candidates are receptacle-centric (empty covered_objects)
    receptacle_centric_hypotheses = {
        item.hypothesis.strip().lower()
        for item in candidates
        if item.hypothesis.strip() and len(item.covered_objects) == 0
    }
    if not open_hypotheses:
        return None

    hypothesis = intent.hypothesis.strip()
    question = intent.question.strip()
    if not hypothesis or not question:
        return None
    if open_hypotheses and hypothesis.lower() not in open_hypotheses:
        return None

    is_receptacle_centric = hypothesis.lower() in receptacle_centric_hypotheses

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
    receptacle = intent.receptacle.strip() if intent.receptacle and intent.receptacle.strip() in state["receptacles"] else None

    # For receptacle-centric candidates, try to extract receptacle from hypothesis if LLM didn't set it
    if is_receptacle_centric and not receptacle:
        for r in state["receptacles"]:
            if r.lower() in hypothesis.lower():
                receptacle = r
                break

    # Allow receptacle-centric candidates (empty covered_objects) if receptacle is set
    if not covered_objects and not receptacle:
        return None

    return PreferenceQuestionIntent(
        question_pattern="preference_eliciting",
        hypothesis=hypothesis,
        covered_objects=covered_objects,
        receptacle=receptacle,
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
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.model: Any = create_chat_model(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
            timeout=120,
        )
        self.structured_model = self.model.with_structured_output(ActionIntent)

    def propose(
        self,
        *,
        state: AgentState,
        guidance: str = "",
    ) -> Optional[ActionIntent]:
        # If no unresolved objects, pick from seen objects not yet asked via AO
        if not state["unresolved_objects"]:
            asked = {qa.get("target", "") for qa in state["qa_history"] if qa.get("question_pattern") == "action_oriented"}
            confirmed = {a.get("object_name", "") for a in state["confirmed_actions"]}
            available = [o for o in state["seen_objects"] if o not in asked and o not in confirmed]
            if not available:
                # All seen objects already confirmed — pick any not yet AO-asked
                available = [o for o in state["seen_objects"] if o not in asked]
            if not available:
                return None
            # Create a working copy with available objects as "unresolved"
            state = {**state, "unresolved_objects": available, "confirmed_actions": []}

        # If guidance mentions boundary/ambiguous/similar, let LLM pick the best object.
        # Otherwise use novelty-based selection.
        strategic_guidance = any(
            w in guidance.lower()
            for w in ["ambiguous", "boundary", "similar", "same", "clustered"]
        )

        if strategic_guidance and len(state["unresolved_objects"]) > 1:
            # Let the LLM choose which object to ask about, based on the guidance
            target_object = self._strategic_select(state=state, guidance=guidance)
        else:
            # Default: novelty-based diversity
            recent_asked = [
                item.get("target", "") or item.get("object_name", "")
                for item in state.get("qa_history", [])[-4:]
                if item.get("question_pattern") == "action_oriented"
            ]
            recent_words = set(" ".join(recent_asked).lower().split()) if recent_asked else set()

            def _novelty(obj: str) -> int:
                obj_words = set(obj.lower().split())
                return -len(obj_words & recent_words)

            target_object = max(state["unresolved_objects"], key=_novelty)

        system_prompt = """
You are a proposer for Action-oriented questions in a household rearrangement task.

Your job:
Ask a direct placement question about the specified target object.

Action mode: always use direct_grounding.

Question rules:
- must mention the target object by name
- ask "Where should the [object] go?" or equivalent
- one sentence, no hedging, no unrelated objects
""".strip()

        user_prompt = f"""
Target object (you MUST ask about this exact object):
{target_object}

Return exactly one ActionIntent:
- question_pattern = "action_oriented"
- action_mode = "direct_grounding"
- object_name = "{target_object}"
- priority = 0.5
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

    def _strategic_select(
        self,
        *,
        state: AgentState,
        guidance: str = "",
    ) -> str:
        """Let LLM pick the best object to ask about based on guidance context.
        Filters out objects that would likely go to already-covered receptacles."""
        objs = state["unresolved_objects"]
        cas = state["confirmed_actions"]
        cps = state["confirmed_preferences"]

        # Pre-filter: exclude objects whose name strongly suggests an already-covered receptacle
        ca_recs = {a.get("receptacle", "") for a in cas}
        cp_recs = {p.get("receptacle", "") for p in cps}
        covered_recs = ca_recs | cp_recs

        prompt = f"""Pick ONE object from the unresolved list to ask about next.

Guidance: {guidance}

Unresolved objects: {objs}

Confirmed actions so far: {[(ca.get('object_name',''), ca.get('receptacle','')) for ca in cas]}

Confirmed preferences: {[(cp.get('hypothesis',''), cp.get('receptacle','')) for cp in cps]}

CRITICAL: Do NOT pick an object that obviously belongs to an already-covered receptacle: {sorted(covered_recs)}.
Pick one that is AMBIGUOUS or likely goes to a DIFFERENT receptacle.

Reply with ONLY the exact object name, nothing else."""

        try:
            llm = create_chat_model(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.0,
                reasoning=False,
                timeout=60,
            )
            resp = llm.invoke(prompt)
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip().strip('"').strip("'")
            for obj in objs:
                if obj.lower() == text.lower():
                    return obj
            for obj in objs:
                if text.lower() in obj.lower() or obj.lower() in text.lower():
                    return obj
        except Exception:
            pass
        return objs[0]


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

        covered_receptacles = sorted({
            cp.get("receptacle") for cp in state["confirmed_preferences"]
            if cp.get("receptacle")
        })
        uncovered_receptacles = sorted({
            r for r in state["receptacles"]
            if r not in set(covered_receptacles)
        })

        user_prompt = f"""
Confirmed actions (ALL accumulated evidence — use the full list, not just recent steps):
{state["confirmed_actions"]}

Seen objects:
{state["seen_objects"]}

Unresolved objects:
{state["unresolved_objects"]}

Confirmed preferences (rules already established):
{state["confirmed_preferences"]}

Receptacles already covered by confirmed preferences:
{covered_receptacles}

Receptacles NOT yet covered — PRIORITIZE these for your hypothesis:
{uncovered_receptacles}

Rejected hypotheses:
{state["negative_preferences"]}

Guidance:
{guidance}

IMPORTANT: Do NOT generate a hypothesis that only addresses receptacles already in "covered".
Look across ALL confirmed_actions to find a pattern that targets one of the uncovered receptacles.
If multiple confirmed_actions point to the same uncovered receptacle, that is strong evidence for a hypothesis.

Each intent must include:
- question_pattern = "preference_induction"
- hypothesis = a concise summary rule to confirm. Describe the object group using shared attributes (material, size, power source, usage context) rather than just a generic type label. Good: "small plug-in bedside electronics go to the nightstand drawer". Bad: "electronics go to the drawer".
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
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
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
