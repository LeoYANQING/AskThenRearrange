import json
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import requests


class Qwen3APIError(RuntimeError):
    pass


class Qwen3API:
    """
    Simple Python interface for local Ollama Qwen3-32B.
    Default host: http://localhost:11434
    """

    def __init__(
        self,
        model: str = "qwen3:32b",
        host: Optional[str] = None,
        timeout: int = 120,
        system_guard: Optional[str] = None,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.system_guard = system_guard or (
            "Respond with only the requested output. Do not include analysis or <think> tags."
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, Iterable[str], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {},
        }
        if system:
            payload["system"] = f"{system}\n{self.system_guard}"
        else:
            payload["system"] = self.system_guard
        return self._post("/api/generate", payload, stream=stream)

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[str, Iterable[str], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options or {},
        }
        if self.system_guard:
            payload["messages"] = (
                [{"role": "system", "content": self.system_guard}] + payload["messages"]
            )
        return self._post("/api/chat", payload, stream=stream)

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Union[str, Iterable[str], Dict[str, Any]]:
        url = f"{self.host}{path}"
        try:
            response = requests.post(url, json=payload, stream=stream, timeout=self.timeout)
        except requests.RequestException as exc:
            raise Qwen3APIError(f"Request failed: {exc}") from exc

        if response.status_code != 200:
            raise Qwen3APIError(f"API {url} failed: {response.status_code} {response.text}")

        if stream:
            return self._iter_stream(response)

        data = response.json()
        if "response" in data:
            return data["response"]
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        return data

    @staticmethod
    def _iter_stream(response: requests.Response) -> Iterable[str]:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in chunk:
                yield chunk["response"]
            elif "message" in chunk and isinstance(chunk["message"], dict):
                content = chunk["message"].get("content", "")
                if content:
                    yield content
            if chunk.get("done"):
                break


if __name__ == "__main__":
    client = Qwen3API()
    reply = client.generate(
        "Hello, please introduce yourself briefly.",
        system="You are a helpful assistant.",
        options={"temperature": 0.7, "num_predict": 512},
    )
    print(reply)
