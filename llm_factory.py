"""Unified LLM factory supporting ollama and OpenAI-compatible backends.

Usage:
  # Ollama mode (default):
  LLM_BACKEND=ollama LLM_MODEL=qwen3 python test_policy_loop.py

  # OpenAI mode:
  LLM_BACKEND=openai LLM_MODEL=gpt-5 LLM_API_KEY=sk-xxx LLM_BASE_URL=https://api.gptsapi.net/v1 python test_policy_loop.py

Environment variables:
  LLM_BACKEND:  "ollama" (default) or "openai"
  LLM_MODEL:    model name (default: "qwen3" for ollama, "gpt-5" for openai)
  LLM_BASE_URL: API base URL
  LLM_API_KEY:  API key (required for openai mode)
"""

import os
from typing import Any

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", os.environ.get("OLLAMA_MODEL", "qwen3"))
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    os.environ.get("OLLAMA_HOST", "http://110.42.252.68:8080"),
)
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


class _OpenAIWrapper:
    """Wrapper that uses JSON mode + manual parsing for structured output.

    Avoids langchain's with_structured_output which has compatibility issues
    with some OpenAI-compatible APIs (empty Dict fields, schema validation errors).
    """
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, *args, **kwargs):
        return self._llm.invoke(*args, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return _JsonModeStructured(self._llm, schema)

    def __getattr__(self, name):
        return getattr(self._llm, name)


class _JsonModeStructured:
    """Invokes LLM with JSON mode and parses output into a pydantic model."""
    def __init__(self, llm, schema):
        self._llm = llm
        self._schema = schema

    def invoke(self, input_val, **kwargs):
        import json as _json
        # Build messages
        if isinstance(input_val, str):
            messages = [{"role": "user", "content": input_val}]
        elif isinstance(input_val, list):
            messages = list(input_val)
        else:
            messages = [{"role": "user", "content": str(input_val)}]

        # Inject JSON instruction into system message
        schema_hint = f"Return ONLY valid JSON matching this schema: {self._schema.model_json_schema()}"
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + schema_hint}
        else:
            messages.insert(0, {"role": "system", "content": schema_hint})

        # Call with JSON mode
        response = self._llm.invoke(
            messages,
            response_format={"type": "json_object"},
        )
        text = response.content if hasattr(response, "content") else str(response)
        try:
            data = _json.loads(text)
            return self._schema.model_validate(data)
        except Exception:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = _json.loads(match.group())
                return self._schema.model_validate(data)
            return self._schema()


def create_chat_model(
    *,
    model: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    timeout: int = 120,
    **kwargs,
) -> Any:
    """Create a chat model instance based on LLM_BACKEND.

    Returns a langchain ChatModel that supports .invoke() and .with_structured_output().
    """
    model = model or LLM_MODEL
    base_url = base_url or LLM_BASE_URL
    backend = LLM_BACKEND

    # Filter out ollama-specific kwargs for openai backend
    kwargs.pop("reasoning", None)

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        api_key = LLM_API_KEY
        if not api_key:
            raise ValueError("LLM_API_KEY is required for openai backend")
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
        return _OpenAIWrapper(llm)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            reasoning=False,
            timeout=timeout,
            **kwargs,
        )


# Convenience: default model name for backward compatibility
DEFAULT_MODEL = LLM_MODEL
DEFAULT_BASE_URL = LLM_BASE_URL
