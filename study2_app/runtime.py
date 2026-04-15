from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from evaluation import FinalPlacementPlanner, finalize_seen_placements, finalize_unseen_placements
from proposers import ActionProposer, PreferenceElicitingProposer, PreferenceInductionProposer
from question_policy import PolicyMode, QuestionDecision, QuestionPolicyController
from state_init import build_initial_state
from state_update import StateUpdate

from study2_app.types import PendingTurnRecord, TurnRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "budget_used": len(state["qa_history"]),
        "unresolved_objects": list(state["unresolved_objects"]),
        "confirmed_actions": list(state["confirmed_actions"]),
        "confirmed_preferences": list(state["confirmed_preferences"]),
        "negative_actions": list(state["negative_actions"]),
        "negative_preferences": list(state["negative_preferences"]),
    }


class Study2Runtime:
    def __init__(
        self,
        *,
        proposer_model: str,
        updater_model: str,
        evaluation_model: str,
        base_url: str,
        budget_total: int,
        selection_method: str,
    ) -> None:
        self.budget_total = budget_total
        self.controller = QuestionPolicyController(
            model=proposer_model,
            base_url=base_url,
            temperature=0.0,
            selection_method=selection_method,  # type: ignore[arg-type]
        )
        self.eliciting_proposer = PreferenceElicitingProposer(
            model=proposer_model,
            base_url=base_url,
            temperature=0.0,
        )
        self.action_proposer = ActionProposer(
            model=proposer_model,
            base_url=base_url,
            temperature=0.0,
        )
        self.induction_proposer = PreferenceInductionProposer(
            model=proposer_model,
            base_url=base_url,
            temperature=0.0,
        )
        self.updater = StateUpdate(
            model=updater_model,
            base_url=base_url,
            temperature=0.0,
        )
        self.planner = FinalPlacementPlanner(
            model=evaluation_model,
            base_url=base_url,
            temperature=0.0,
        )

    def initialize_state(self, episode: Any) -> dict[str, Any]:
        return build_initial_state(
            episode=episode,
            strategy="parallel_exploration",
            budget_total=self.budget_total,
        )

    def _propose_intent(self, *, state: dict[str, Any], decision: QuestionDecision) -> Any:
        if decision.question_pattern == "preference_eliciting":
            return self.eliciting_proposer.propose(
                state=state,
                guidance=decision.guidance,
            )
        if decision.question_pattern == "action_oriented":
            return self.action_proposer.propose(
                state=state,
                guidance=decision.guidance,
            )
        if decision.question_pattern == "preference_induction":
            intents = self.induction_proposer.propose(
                state=state,
                max_intents=3,
                guidance=decision.guidance,
            )
            return intents[0] if intents else None
        raise ValueError(f"Unsupported question pattern: {decision.question_pattern}")

    def prepare_next_turn(
        self,
        *,
        state: dict[str, Any],
        mode: PolicyMode,
    ) -> PendingTurnRecord | None:
        if len(state["qa_history"]) >= state["budget_total"]:
            return None

        decision = self.controller.plan_next_question(state=state, mode=mode)
        if decision is None:
            return None

        intent = self._propose_intent(state=state, decision=decision)
        if intent is None and decision.question_pattern == "preference_induction":
            decision = QuestionDecision(
                question_pattern="action_oriented",
                guidance="No induction pattern available; ask a direct placement question to gather more evidence.",
            )
            intent = self._propose_intent(state=state, decision=decision)

        if intent is None:
            return None

        question = intent.question if hasattr(intent, "question") else str(intent.get("question", ""))
        if not question:
            raise RuntimeError("Question generation returned an empty question.")

        if hasattr(intent, "object_name"):
            target = intent.object_name
            action_mode = intent.action_mode
            hypothesis = ""
            covered_objects: list[str] = []
            receptacle = None
        else:
            target = str(intent.get("hypothesis", "")) or str(intent.get("target", ""))
            action_mode = intent.get("action_mode")
            hypothesis = str(intent.get("hypothesis", ""))
            covered_objects = list(intent.get("covered_objects", []))
            receptacle = intent.get("receptacle")

        return PendingTurnRecord(
            turn_index=len(state["qa_history"]) + 1,
            question_pattern=decision.question_pattern,
            guidance=decision.guidance,
            target=target,
            question=question,
            action_mode=action_mode,
            hypothesis=hypothesis,
            covered_objects=covered_objects,
            receptacle=receptacle,
            created_at=utc_now(),
            retry_count=0,
        )

    def apply_answer_and_advance(
        self,
        *,
        state: dict[str, Any],
        mode: PolicyMode,
        pending_turn: PendingTurnRecord,
        answer_text: str,
        retry_count: int = 0,
    ) -> tuple[dict[str, Any], TurnRecord, PendingTurnRecord | None]:
        next_state = copy.deepcopy(state)
        pattern = pending_turn["question_pattern"]

        if pattern == "action_oriented":
            next_state = self.updater.update_state_from_action_answer(
                state=next_state,
                target=pending_turn["target"],
                answer=answer_text,
                question=pending_turn["question"],
                action_mode=pending_turn["action_mode"],
            )
        elif pattern == "preference_eliciting":
            next_state = self.updater.update_state_from_preference_eliciting_answer(
                state=next_state,
                hypothesis=pending_turn["hypothesis"],
                covered_objects=pending_turn["covered_objects"],
                answer=answer_text,
                question=pending_turn["question"],
                oracle_receptacle=pending_turn["receptacle"],
            )
        elif pattern == "preference_induction":
            next_state = self.updater.update_state_from_preference_induction_answer(
                state=next_state,
                hypothesis=pending_turn["hypothesis"],
                covered_objects=pending_turn["covered_objects"],
                answer=answer_text,
                question=pending_turn["question"],
            )
        else:
            raise ValueError(f"Unsupported pending turn pattern: {pattern}")

        turn_record = TurnRecord(
            turn_index=pending_turn["turn_index"],
            question_pattern=pattern,
            guidance=pending_turn["guidance"],
            target=pending_turn["target"],
            question=pending_turn["question"],
            action_mode=pending_turn["action_mode"],
            covered_objects=list(pending_turn["covered_objects"]),
            participant_answer_raw=answer_text,
            update_status="applied",
            retry_count=retry_count,
            state_summary=state_summary(next_state),
            created_at=pending_turn["created_at"],
            answered_at=utc_now(),
        )

        if len(next_state["qa_history"]) >= next_state["budget_total"]:
            return next_state, turn_record, None

        next_pending = self.prepare_next_turn(state=next_state, mode=mode)
        return next_state, turn_record, next_pending

    def finalize_trial(self, *, state: dict[str, Any]) -> dict[str, str]:
        seen = finalize_seen_placements(state, planner=self.planner)
        unseen = finalize_unseen_placements(state, planner=self.planner)
        return {**seen, **unseen}
