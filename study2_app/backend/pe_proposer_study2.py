"""Study-2-only Preference-Eliciting proposer.

Subclasses the Study 1 `PreferenceElicitingProposer` and overrides only the
selection bias — all candidate construction, model plumbing, and output
normalization are inherited unchanged so Study 1 behavior in `proposers.py`
stays identical.

Bias fixes (why Study 2 participants saw 88% receptacle-centric questions):

  1. Candidate ordering is category-first. Parent prepends RC candidates to
     the list; combined with LLM position bias this makes RC almost always
     win. We place category candidates first.

  2. Selection prompt is neutral. Parent tells the LLM to "Prefer
     receptacle-centric". We remove that bias and instruct an information-
     value-based choice. RC phrasing and covered_objects constraints are
     kept so the downstream normalization path still works.

Not changed:
  - Candidate pools (_build_preference_candidates, _build_receptacle_centric_candidates)
  - Intent schema, structured model plumbing, _normalize_preference_eliciting_intent
  - Any parent method that doesn't involve selection bias
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proposers import (  # noqa: E402
    BuiltPreferenceCandidateModel,
    PreferenceElicitingProposer,
    PreferenceQuestionIntent,
    _normalize_preference_eliciting_intent,
)
from agent_schema import AgentState  # noqa: E402


_CAT_REWRITE_TEMPLATE = "How do you usually organize {hyp}, like {examples}?"


def _format_examples(covered: List[str], max_n: int = 2) -> str:
    picks = [c for c in covered if c][:max_n]
    if not picks:
        return ""
    if len(picks) == 1:
        return picks[0]
    return " or ".join(picks)

_HYP_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "items", "item", "things", "thing", "kept", "typically", "is", "are",
    "that", "which", "some", "such",
}


def _hypothesis_keywords(hypothesis: str) -> set:
    words = re.findall(r"[a-z]+", hypothesis.lower())
    return {w for w in words if len(w) >= 3 and w not in _HYP_STOPWORDS}


def _question_aligns_with_hypothesis(question: str, hypothesis: str) -> bool:
    kws = _hypothesis_keywords(hypothesis)
    if not kws:
        return True
    q_lower = question.lower()
    return any(kw in q_lower for kw in kws)


def _count_receptacle_mentions(text: str, receptacles: List[str]) -> int:
    t = text.lower()
    return sum(1 for r in receptacles if r.lower() in t)


def _is_rc_candidate(candidate: BuiltPreferenceCandidateModel) -> bool:
    return len(candidate.covered_objects) == 0


def _drop_compound_scope_candidates(
    candidates: List[BuiltPreferenceCandidateModel],
    receptacles: List[str],
) -> List[BuiltPreferenceCandidateModel]:
    """Reject candidates that mix two receptacle scopes in a single hypothesis.

    A CAT candidate whose hypothesis names two receptacles (e.g., "plug-in
    countertop appliances and drawer items") would produce a question asking
    the user to confirm a rule that spans two receptacles — which is hard to
    answer and violates the one-class-per-question contract.
    RC candidates inherently name one receptacle, so they are kept unchanged.
    """
    kept = []
    for c in candidates:
        if _is_rc_candidate(c):
            kept.append(c)
            continue
        if _count_receptacle_mentions(c.hypothesis, receptacles) >= 2:
            continue
        kept.append(c)
    return kept


class Study2PreferenceElicitingProposer(PreferenceElicitingProposer):
    def propose(
        self,
        *,
        state: AgentState,
        guidance: str = "",
        max_candidates: int = 5,
    ) -> Optional[PreferenceQuestionIntent]:
        # Turn-0 fixed general-preference probe (UPF): skip all candidate
        # selection and ask a room-level open question. All seen_objects are
        # marked as covered so state_update can extract whatever specific
        # placements the user mentions in their free-form answer.
        n_turns = len(state.get("qa_history", []))
        if n_turns == 0:
            room = state.get("room") or "home"
            return PreferenceQuestionIntent(
                question_pattern="preference_eliciting",
                hypothesis=f"general organization of the {room}",
                covered_objects=list(state.get("seen_objects", [])),
                receptacle=None,
                priority=1.0,
                question=f"How do you usually like to organize your {room}?",
            )

        category_candidates = self._build_preference_candidates(
            state=state,
            max_candidates=max_candidates,
        )
        rc_candidates = self._build_receptacle_centric_candidates(
            state=state,
            max_candidates=2,
        )

        # One-class-per-question: drop CAT candidates whose hypothesis
        # references two or more receptacles (e.g. "plug-in countertop
        # appliances and drawer items"). Those candidates ask the user to
        # confirm a compound rule that mixes CAT + RC scopes in one turn.
        receptacles = state["receptacles"]
        category_candidates = _drop_compound_scope_candidates(category_candidates, receptacles)

        # Early-turn CAT preference: when a strong category candidate exists
        # in the first two turns, strip RC from the pool so the selector
        # cannot fall back to the receptacle path before categories get a
        # real shot. With the unrestricted pool, 90% of picks went RC
        # because RC candidates vastly outnumber strong categories early on.
        n_turns = len(state.get("qa_history", []))
        strong_cat_exists = any(len(c.covered_objects) >= 2 for c in category_candidates)
        if n_turns < 2 and strong_cat_exists:
            candidates = category_candidates
        else:
            candidates = category_candidates + rc_candidates

        intent = self._propose_from_candidates_neutral(
            state=state,
            candidates=candidates,
            guidance=guidance,
        )

        # Alignment post-check: if the selector picked a CAT candidate
        # (non-empty covered_objects) but generated a question whose text
        # doesn't reference the hypothesis, it confused itself (observed
        # as "hyp=warm and cozy textile coverings" paired with a question
        # about books/magazines or a single object). Rewrite with a
        # canonical CAT template so the participant always hears a
        # question about the actual category.
        if intent is not None and intent.get("covered_objects"):
            if not _question_aligns_with_hypothesis(intent["question"], intent["hypothesis"]):
                examples = _format_examples(intent["covered_objects"])
                intent["question"] = _CAT_REWRITE_TEMPLATE.format(
                    hyp=intent["hypothesis"], examples=examples
                )

        # One-class-per-question post-check: even with a clean CAT hypothesis,
        # the LLM sometimes inserts two receptacle names into the question
        # text (e.g. "plug-in countertop appliances and drawer items" →
        # participant gets asked about countertop AND drawer in one breath).
        # If the final question text mentions ≥2 receptacle names, rewrite
        # using the canonical CAT template which references only the
        # hypothesis — no receptacles.
        if intent is not None and intent.get("covered_objects"):
            if _count_receptacle_mentions(intent["question"], receptacles) >= 2:
                examples = _format_examples(intent["covered_objects"])
                intent["question"] = _CAT_REWRITE_TEMPLATE.format(
                    hyp=intent["hypothesis"], examples=examples
                )

        return intent

    def _propose_from_candidates_neutral(
        self,
        *,
        state: AgentState,
        candidates: List[BuiltPreferenceCandidateModel],
        guidance: str = "",
    ) -> Optional[PreferenceQuestionIntent]:
        if not candidates:
            return None

        system_prompt = """
Choose one preference-eliciting candidate. Pick the one with highest info value.

Question format — always a HOW/principle-style open question about organization habits:
- Category candidate (non-empty covered_objects): "How do you usually organize [category], like [example1] or [example2]?"
  Pick 1–2 examples verbatim from that candidate's covered_objects.
- Receptacle candidate (empty covered_objects): "What kinds of items do you typically keep in the [receptacle]?"

Hard rules:
- Use the hypothesis exactly as given.
- Exactly ONE category OR ONE receptacle per question — never two.
- No yes/no questions. Ask how/what-kind — not "where do you put X?".
""".strip()

        covered_receptacles = sorted({
            cp.get("receptacle") for cp in state["confirmed_preferences"] if cp.get("receptacle")
        })
        uncovered_receptacles = sorted({
            r for r in state["receptacles"] if r not in set(covered_receptacles)
        })

        user_prompt = f"""
Candidates:
{candidates}

Unresolved objects: {state["unresolved_objects"]}
Uncovered receptacles: {uncovered_receptacles}
Guidance: {guidance}

Return one intent.

Fields:
- hypothesis = one hypothesis verbatim from a candidate
- covered_objects = verbatim subset from that candidate ([] if receptacle-centric)
- receptacle = best-guess exact name from the receptacles list
- priority = 0.0 to 1.0
- question = HOW/principle-style open question. Category: "How do you usually organize [hypothesis], like [example1] or [example2]?" (pick 1-2 verbatim examples from covered_objects). Receptacle: "What kinds of items do you typically keep in the [receptacle]?"
""".strip()

        result = self.structured_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return _normalize_preference_eliciting_intent(
            intent=result,
            state=state,
            candidates=candidates,
        )
