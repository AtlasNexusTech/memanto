"""
Tests for the LocalLLMClient and its soft-fail integration.

Uses a fake Ollama-compatible server (no real LLM needed).
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from memanto.app.backends.llm_client import LocalLLMClient, LocalLLMError


class FakeOllamaHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible fake: /api/tags and /v1/chat/completions."""

    responses = {}

    def do_GET(self):  # /api/tags
        if self.path == "/api/tags":
            self._send_json(200, {"models": [{"name": "fake"}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):  # /v1/chat/completions
        if self.path == "/v1/chat/completions":
            self._send_json(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": FakeOllamaHandler.responses.get(
                                    "content", "Fake answer"
                                )
                            }
                        }
                    ]
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, code, payload):
        import json

        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def fake_ollama():
    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    FakeOllamaHandler.responses = {"content": "Le budget est de 5000 euros."}
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def dead_ollama():
    return "http://127.0.0.1:1"  # nothing listens on port 1


def test_is_available_true(fake_ollama):
    client = LocalLLMClient(base_url=fake_ollama)
    assert client.is_available() is True


def test_is_available_false(dead_ollama):
    client = LocalLLMClient(base_url=dead_ollama)
    assert client.is_available() is False


def test_chat_returns_content(fake_ollama):
    client = LocalLLMClient(base_url=fake_ollama)
    out = client.chat(system="s", user="u")
    assert out == "Le budget est de 5000 euros."


def test_chat_raises_when_unreachable(dead_ollama):
    client = LocalLLMClient(base_url=dead_ollama, timeout=0.5)
    with pytest.raises(LocalLLMError):
        client.chat(system="s", user="u")


def test_generate_answer_uses_context(fake_ollama):
    client = LocalLLMClient(base_url=fake_ollama)
    out = client.generate_answer(
        "quel est le budget",
        [{"id": "m2", "text": "[PREFERENCE] Budget\n\n5000 euros."}],
        header_prompt="system",
        footer_prompt="answer now",
    )
    assert "5000" in out


def test_check_contradiction_parses_json(fake_ollama):
    FakeOllamaHandler.responses = {
        "content": '{"contradicts": true, "conflicting_ids": ["m1"], "reason": "opposes email preference"}'
    }
    client = LocalLLMClient(base_url=fake_ollama)
    result = client.check_contradiction(
        "Le client prefere les appels.",
        [{"id": "m1", "text": "[PREFERENCE] Le client prefere l'email."}],
    )
    assert result["contradicts"] is True
    assert result["conflicting_ids"] == ["m1"]


def test_check_contradiction_fails_soft_on_bad_json(fake_ollama):
    FakeOllamaHandler.responses = {"content": "not json at all"}
    client = LocalLLMClient(base_url=fake_ollama)
    result = client.check_contradiction("x", [{"id": "m1", "text": "y"}])
    assert result == {"contradicts": False, "conflicting_ids": [], "reason": ""}
