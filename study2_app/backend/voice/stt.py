"""STT via Aliyun Dashscope (paraformer-realtime-v2).

Accepts an audio file (wav/mp3/pcm/opus/etc.), one-shot recognizes it, returns text.
Requires env var ``DASHSCOPE_API_KEY``.
"""
from __future__ import annotations

import os
from typing import Any

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY", "sk-515fc7843e934051bc2d59978fc9e030"
)
dashscope.api_key = DASHSCOPE_API_KEY


class _NoopCallback(RecognitionCallback):
    def on_open(self) -> None: ...
    def on_close(self) -> None: ...
    def on_complete(self) -> None: ...
    def on_error(self, result) -> None: ...
    def on_event(self, result) -> None: ...


def _ensure_api_key() -> None:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY not set.")


def transcribe_file(
    file_path: str,
    *,
    audio_format: str = "wav",
    sample_rate: int = 16000,
    language_hints: list[str] | None = None,
    model: str = "paraformer-realtime-v2",
) -> dict[str, Any]:
    """One-shot recognize a local audio file. Returns ``{"text": str, "raw": list}``.

    ``language_hints`` narrows recognition (e.g. ``["zh", "en"]``).
    """
    _ensure_api_key()
    kwargs: dict[str, Any] = {}
    if language_hints:
        kwargs["language_hints"] = language_hints

    recognition = Recognition(
        model=model,
        callback=_NoopCallback(),
        format=audio_format,
        sample_rate=sample_rate,
        **kwargs,
    )
    result = recognition.call(file=file_path)
    sentences = result.get_sentence() or []
    text = "".join((s.get("text") or "") for s in sentences).strip()
    return {"text": text, "raw": sentences}
