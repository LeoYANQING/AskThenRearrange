from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from study2_app.backend.session_store import LOGS_DIR, get_session

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/{session_id}")
def download_log(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    log_path = LOGS_DIR / f"{session_id}.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    return FileResponse(
        path=str(log_path),
        media_type="application/x-ndjson",
        filename=f"session_{session_id}.jsonl",
    )


@router.get("/{session_id}/text")
def view_log(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    log_path = LOGS_DIR / f"{session_id}.jsonl"
    if not log_path.exists():
        return PlainTextResponse("(no log entries yet)")

    return PlainTextResponse(log_path.read_text(encoding="utf-8"))
