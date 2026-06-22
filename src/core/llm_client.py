from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.core.config import LLMConfig


class LocalLLMClient:
    """Cliente mínimo para APIs locais compatíveis com OpenAI Chat Completions."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "response_format": {"type": "json_object"},
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "pipeline_output", "schema": schema},
            }

        key = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["parsed"]

        raw = self._request(payload)
        content = raw["choices"][0]["message"].get("content") or "{}"
        parsed = self._parse_json(content)
        usage = raw.get("usage", {})
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        cache_path.write_text(
            json.dumps({"request": payload, "response": raw, "parsed": parsed}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return parsed

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.config.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Falha ao chamar LM local em {endpoint}: {last_error}")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("A resposta JSON do LM deve ser um objeto")
        return parsed
