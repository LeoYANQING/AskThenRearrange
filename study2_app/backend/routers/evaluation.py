from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from study2_app.backend.models import PreferenceFormInput, ScoreResponse, SessionSnapshot
from study2_app.backend.session_store import get_session

router = APIRouter(prefix="/sessions", tags=["evaluation"])


@router.post("/{session_id}/preference_form", response_model=SessionSnapshot)
def submit_preference_form(session_id: str, body: PreferenceFormInput):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.phase not in ("dialogue_complete",):
        raise HTTPException(
            status_code=400,
            detail=f"Not in dialogue_complete phase (current: {session.phase})",
        )

    trial = session.current_trial_snapshot()
    if trial is None:
        raise HTTPException(status_code=400, detail="No active trial")

    trial["preference_assignments"] = dict(body.assignments)
    trial["phase"] = "preference_form"
    session.phase = "preference_form"

    session.log_event("preference_form_submitted", {"assignments": body.assignments})
    return session.to_snapshot()


@router.post("/{session_id}/score", response_model=ScoreResponse)
def compute_score(session_id: str, request: Request):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.phase not in ("preference_form",):
        raise HTTPException(
            status_code=400,
            detail=f"Not in preference_form phase (current: {session.phase})",
        )

    state = session.agent_state
    episode = session.current_episode
    if state is None or episode is None:
        raise HTTPException(status_code=400, detail="No active trial data")

    trial = session.current_trial_snapshot()
    participant_gold = (trial or {}).get("preference_assignments") or {}
    if not participant_gold:
        raise HTTPException(status_code=400, detail="No preference form submitted")

    planner = request.app.state.planner

    from evaluation import (  # noqa: E402
        finalize_seen_placements,
        finalize_unseen_placements,
        placement_accuracy,
    )

    predicted_seen = finalize_seen_placements(state, planner=planner)
    predicted_unseen = finalize_unseen_placements(state, planner=planner)

    all_predicted = {**predicted_seen, **predicted_unseen}

    # Study 2 uses the participant's post-dialogue preference form as ground
    # truth (not the Study 1 canonical placements in episode.*_placements).
    gold_seen = {o: participant_gold[o] for o in episode.seen_objects if o in participant_gold}
    gold_unseen = {o: participant_gold[o] for o in episode.unseen_objects if o in participant_gold}

    seen_psr = placement_accuracy(predicted_seen, gold_seen, episode.seen_objects)
    unseen_psr = placement_accuracy(predicted_unseen, gold_unseen, episode.unseen_objects)
    total_objects = episode.seen_objects + episode.unseen_objects
    total_psr = placement_accuracy(all_predicted, participant_gold, total_objects)

    item_scores: dict = {}
    for obj in episode.seen_objects:
        item_scores[obj] = all_predicted.get(obj) == gold_seen.get(obj)
    for obj in episode.unseen_objects:
        item_scores[obj] = all_predicted.get(obj) == gold_unseen.get(obj)

    trial = session.current_trial_snapshot()
    if trial is not None:
        trial["predicted_placements"] = all_predicted
        trial["psr"] = {
            "seen_psr": seen_psr,
            "unseen_psr": unseen_psr,
            "total_psr": total_psr,
            "item_scores": item_scores,
        }
        trial["phase"] = "prediction_done"

    session.log_event("score_computed", {
        "episode_id": episode.episode_id,
        "seen_psr": seen_psr,
        "unseen_psr": unseen_psr,
        "total_psr": total_psr,
        "predicted_placements": dict(all_predicted),
        "participant_gold": dict(participant_gold),
        "item_scores": dict(item_scores),
    })

    # Advance to next trial or final ranking
    session.current_trial_index += 1
    if session.current_trial_index >= 3:
        session.phase = "final_ranking"
        session.log_event("all_trials_complete", {})
    else:
        session.phase = "created"  # ready for next trial
        session.agent_state = None
        session.current_episode = None
        session.pending_qa = None

    return ScoreResponse(
        seen_psr=seen_psr,
        unseen_psr=unseen_psr,
        total_psr=total_psr,
        item_scores=item_scores,
    )
