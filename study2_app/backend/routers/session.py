from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from study2_app.backend.models import (
    CreateSessionInput,
    FinalInput,
    SessionSnapshot,
    TrialStartInput,
)
from study2_app.backend.session_store import (
    EPISODES_BY_ROOM,
    create_session,
    get_episode_for_room,
    get_session,
)

router = APIRouter(prefix="/sessions", tags=["session"])


@router.post("", response_model=SessionSnapshot)
def create_session_endpoint(body: CreateSessionInput):
    try:
        session = create_session(
            participant_id=body.participant_id,
            latin_square_row=body.latin_square_row,
            notes=body.notes,
            budget_total=body.budget_total,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return session.to_snapshot()


@router.get("/{session_id}", response_model=SessionSnapshot)
def get_session_endpoint(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_snapshot()


@router.post("/{session_id}/trial", response_model=SessionSnapshot)
def start_trial(session_id: str, body: TrialStartInput):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.current_trial_index >= 3:
        raise HTTPException(status_code=400, detail="All trials already completed")

    expected_config = session.current_trial_config()
    if body.room_type and body.room_type != expected_config["room_type"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"room_type mismatch: Latin square expects '{expected_config['room_type']}', "
                f"got '{body.room_type}'"
            ),
        )

    available = len(EPISODES_BY_ROOM.get(expected_config["room_type"], []))
    if not 0 <= body.episode_index < available:
        raise HTTPException(
            status_code=400,
            detail=f"episode_index must be 0–{available - 1} for '{expected_config['room_type']}'",
        )

    # Build Chinese-display mapping for this episode (Study 2 only).
    # Backend state and scoring remain English; this map is used purely at the
    # dialogue API boundary and for frontend tag labels.
    from study2_app.backend.translate import build_name_mapping
    episode = get_episode_for_room(expected_config["room_type"], body.episode_index)
    names = list(episode.receptacles) + list(episode.seen_objects) + list(episode.unseen_objects)
    name_mapping = build_name_mapping(names)

    try:
        session.start_trial(body.episode_index, name_mapping=name_mapping)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return session.to_snapshot()


@router.post("/{session_id}/final_ranking", response_model=SessionSnapshot)
def submit_final_ranking(session_id: str, body: FinalInput):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.phase not in ("final_ranking", "prediction_done"):
        raise HTTPException(
            status_code=400,
            detail=f"Not in final_ranking phase (current: {session.phase})",
        )

    session.phase = "completed"
    session.strategy_ranking = list(body.strategy_ranking)
    session.final_comment = body.comment
    session.log_event("final_ranking_submitted", {
        "strategy_ranking": body.strategy_ranking,
        "comment": body.comment,
    })
    return session.to_snapshot()
