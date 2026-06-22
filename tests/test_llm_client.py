import json

from src.core.config import LLMConfig
from src.core.llm_client import LocalLLMClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }
        ).encode()


def test_openai_compatible_client_and_cache(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setattr("src.core.llm_client.urlopen", fake_urlopen)
    client = LocalLLMClient(
        LLMConfig(enabled=True, cache_dir=str(tmp_path), base_url="http://local/v1")
    )
    messages = [{"role": "user", "content": "teste"}]
    assert client.chat_json(messages) == {"ok": True}
    assert client.chat_json(messages) == {"ok": True}
    assert calls == [("http://local/v1/chat/completions", 120)]
    assert client.usage["requests"] == 1
