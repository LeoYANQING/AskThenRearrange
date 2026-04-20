"""Voice endpoints — STT via Dashscope paraformer."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from study2_app.backend.voice.stt import transcribe_file

router = APIRouter(prefix="/voice", tags=["voice"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "stt_backend": "dashscope.paraformer-realtime-v2",
        "api_key_set": bool(os.environ.get("DASHSCOPE_API_KEY")),
    }


LANG_MAP = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "auto": None}


@router.post("/stt")
async def stt(
    file: UploadFile = File(...),
    language: Literal["zh", "en", "ja", "ko", "auto"] = Form("zh"),
    sample_rate: int = Form(16000),
):
    if not file.content_type or not file.content_type.startswith("audio"):
        raise HTTPException(400, f"Expected audio, got {file.content_type}")
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio")

    fmt = suffix.lstrip(".")
    if fmt not in {"wav", "mp3", "pcm", "opus", "aac", "amr"}:
        fmt = "wav"

    lang_hint = LANG_MAP.get(language)
    hints = [lang_hint] if lang_hint else None

    def _run() -> dict:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return transcribe_file(
                tmp_path,
                audio_format=fmt,
                sample_rate=sample_rate,
                language_hints=hints,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    try:
        return await asyncio.to_thread(_run)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(502, f"Dashscope STT error: {e}")
