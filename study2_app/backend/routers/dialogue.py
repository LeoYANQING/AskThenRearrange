from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from study2_app.backend.models import NextQuestionResponse, SessionSnapshot, SubmitAnswerInput
from study2_app.backend.session_store import (
    STRATEGY_TO_MODE,
    PendingQA,
    get_session,
)
from study2_app.backend.translate import translate_en_to_zh, translate_zh_to_en

router = APIRouter(prefix="/dialogue", tags=["dialogue"])


def _log_question_generated(session, result: "NextQuestionResponse") -> None:
    pending = session.pending_qa
    session.log_event("question_generated", {
        "pattern": result.pattern,
        "turn_index": result.turn_index,
        "question_zh": result.question,
        "question_en": pending.question if pending else result.question,
        "target": pending.target if pending else None,
        "covered_objects": (pending.covered_objects if pending else None) or [],
        "action_mode": pending.action_mode if pending else None,
    })


def _generate_next_question(session, request: Request) -> NextQuestionResponse | None:
    """Call the policy + proposer to generate the next question. Returns None if dialogue is done."""
    state = session.agent_state
    if len(state.get("unresolved_objects", [])) == 0:
        return None
    config = session.current_trial_config()
    mode = STRATEGY_TO_MODE[config["strategy"]]

    policy = request.app.state.policy
    decision = policy.plan_next_question(state=state, mode=mode)
    if decision is None:
        return None

    pattern = decision.question_pattern
    guidance = decision.guidance
    turn_index = len(state["qa_history"])

    if pattern == "action_oriented":
        proposer = request.app.state.ao_proposer
        intent = proposer.propose(state=state, guidance=guidance)
        if intent is None:
            return None
        pending = PendingQA(
            pattern="action_oriented",
            question=intent.question,
            target=intent.object_name,
            action_mode=intent.action_mode,
            turn_index=turn_index,
        )

    elif pattern == "preference_eliciting":
        proposer = request.app.state.pe_proposer
        intent = proposer.propose(state=state, guidance=guidance)
        if intent is None:
            # Fallback to AO
            ao = request.app.state.ao_proposer
            intent = ao.propose(state=state, guidance=guidance)
            if intent is None:
                return None
            pending = PendingQA(
                pattern="action_oriented",
                question=intent.question,
                target=intent.object_name,
                action_mode=intent.action_mode,
                turn_index=turn_index,
            )
        else:
            pending = PendingQA(
                pattern="preference_eliciting",
                question=intent["question"],
                target=intent["hypothesis"],
                covered_objects=list(intent.get("covered_objects") or []),
                turn_index=turn_index,
            )

    elif pattern == "preference_induction":
        proposer = request.app.state.pi_proposer
        intents = proposer.propose(state=state, guidance=guidance)
        if not intents:
            # Fallback to AO
            ao = request.app.state.ao_proposer
            intent = ao.propose(state=state, guidance=guidance)
            if intent is None:
                return None
            pending = PendingQA(
                pattern="action_oriented",
                question=intent.question,
                target=intent.object_name,
                action_mode=intent.action_mode,
                turn_index=turn_index,
            )
        else:
            best = intents[0]
            covered = list(best.get("covered_objects") or [])
            q = (best.get("question") or "").strip()
            hyp = (best.get("hypothesis") or "").strip()
            # Guard against structured-output failure where the LLM fills
            # `question` with the pattern name or a near-empty token.
            if not q or q.lower() in {"preference_induction", "preference induction", hyp.lower()} or len(q) < 8:
                q = f"Does this summarize your preference: {hyp}?" if hyp else ""
            if not q:
                ao = request.app.state.ao_proposer
                intent = ao.propose(state=state, guidance=guidance)
                if intent is None:
                    return None
                pending = PendingQA(
                    pattern="action_oriented",
                    question=intent.question,
                    target=intent.object_name,
                    action_mode=intent.action_mode,
                    turn_index=turn_index,
                )
            else:
                pending = PendingQA(
                    pattern="preference_induction",
                    question=q,
                    target=hyp,
                    covered_objects=covered,
                    turn_index=turn_index,
                )
    else:
        return None

    # Translate the English question to Chinese for display. Keep the English
    # original on pending.question so state_update sees exactly what Study 1
    # would see.
    trial = session.current_trial_snapshot()
    name_map = (trial or {}).get("name_mapping", {})
    pending.question_zh = translate_en_to_zh(pending.question, name_map)

    session.pending_qa = pending
    return NextQuestionResponse(
        question=pending.question_zh or pending.question,
        pattern=pending.pattern,
        turn_index=pending.turn_index,
        dialogue_complete=False,
    )


@router.post("/{session_id}/start", response_model=NextQuestionResponse)
def start_dialogue(session_id: str, request: Request):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.phase not in ("scene_intro",):
        raise HTTPException(status_code=400, detail=f"Not in scene_intro phase (current: {session.phase})")
    if session.agent_state is None:
        raise HTTPException(status_code=400, detail="No trial loaded — call POST /sessions/{id}/trial first")

    session.phase = "dialogue"
    trial = session.current_trial_snapshot()
    if trial:
        trial["phase"] = "dialogue"

    session.log_event("dialogue_started", {})

    result = _generate_next_question(session, request)
    if result is None:
        session.phase = "dialogue_complete"
        reason = "all_resolved" if len(session.agent_state.get("unresolved_objects", [])) == 0 else "auto_satisfied"
        if trial:
            trial["phase"] = "dialogue_complete"
            trial["stop_reason"] = reason
            trial["turns_used"] = len(session.agent_state["qa_history"])
        session.log_event("dialogue_complete", {
            "stop_reason": reason,
            "turns_used": len(session.agent_state["qa_history"]),
            "state_final": session.agent_state_snapshot(),
        })
        return NextQuestionResponse(question="", pattern="", turn_index=0, dialogue_complete=True)

    _log_question_generated(session, result)
    return result


@router.post("/{session_id}/answer", response_model=NextQuestionResponse)
def submit_answer(session_id: str, body: SubmitAnswerInput, request: Request):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.phase != "dialogue":
        raise HTTPException(status_code=400, detail=f"Not in dialogue phase (current: {session.phase})")
    if session.pending_qa is None:
        raise HTTPException(status_code=400, detail="No pending question — call /start first")

    pending = session.pending_qa
    state = session.agent_state
    updater = request.app.state.state_updater
    answer_zh = body.answer.strip()

    # Translate Chinese answer to English before feeding state_update, so the
    # Study 1 pipeline sees English names that match the episode data.
    trial = session.current_trial_snapshot()
    name_map = (trial or {}).get("name_mapping", {})
    answer_en = translate_zh_to_en(answer_zh, name_map)

    # Update state based on pattern
    if pending.pattern == "action_oriented":
        updater.update_state_from_action_answer(
            state=state,
            target=pending.target,
            answer=answer_en,
            question=pending.question,
            action_mode=pending.action_mode,
        )
    elif pending.pattern == "preference_eliciting":
        updater.update_state_from_preference_eliciting_answer(
            state=state,
            hypothesis=pending.target,
            covered_objects=pending.covered_objects or None,
            answer=answer_en,
            question=pending.question,
            oracle_receptacle=None,
        )
    elif pending.pattern == "preference_induction":
        updater.update_state_from_preference_induction_answer(
            state=state,
            hypothesis=pending.target,
            covered_objects=pending.covered_objects or None,
            answer=answer_en,
            question=pending.question,
        )

    # Record completed turn — save both the Chinese as shown to the participant
    # and the English as consumed by state_update, for full traceability.
    if trial is not None:
        trial["dialogue"].append({
            "turn_index": pending.turn_index,
            "pattern": pending.pattern,
            "question": pending.question_zh or pending.question,
            "answer": answer_zh,
            "question_en": pending.question,
            "answer_en": answer_en,
            "state_after": session.agent_state_snapshot(),
        })

    session.log_event("answer_submitted", {
        "turn_index": pending.turn_index,
        "pattern": pending.pattern,
        "question_zh": pending.question_zh or pending.question,
        "question_en": pending.question,
        "target": pending.target,
        "covered_objects": list(pending.covered_objects or []),
        "action_mode": pending.action_mode,
        "answer_zh": answer_zh,
        "answer_en": answer_en,
        "turns_used": len(state["qa_history"]),
        "state_after": session.agent_state_snapshot(),
    })

    session.pending_qa = None

    # Generate next question
    result = _generate_next_question(session, request)
    if result is None:
        session.phase = "dialogue_complete"
        reason = "all_resolved" if len(state.get("unresolved_objects", [])) == 0 else "auto_satisfied"
        if trial:
            trial["phase"] = "dialogue_complete"
            trial["stop_reason"] = reason
            trial["turns_used"] = len(state["qa_history"])
        session.log_event("dialogue_complete", {
            "stop_reason": reason,
            "turns_used": len(state["qa_history"]),
            "state_final": session.agent_state_snapshot(),
        })
        return NextQuestionResponse(question="", pattern="", turn_index=0, dialogue_complete=True)

    _log_question_generated(session, result)
    return result


@router.post("/{session_id}/stop", response_model=SessionSnapshot)
def stop_dialogue(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.phase != "dialogue":
        raise HTTPException(status_code=400, detail=f"Not in dialogue phase (current: {session.phase})")

    state = session.agent_state
    turns = len(state["qa_history"]) if state else 0

    session.phase = "dialogue_complete"
    trial = session.current_trial_snapshot()
    if trial:
        trial["phase"] = "dialogue_complete"
        trial["stop_reason"] = "user_terminated"
        trial["turns_used"] = turns

    session.pending_qa = None
    session.log_event("dialogue_stopped", {
        "stop_reason": "user_terminated",
        "turns_used": turns,
        "state_final": session.agent_state_snapshot(),
    })
    return session.to_snapshot()
