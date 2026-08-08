"""
LocalLLMClient — optional local LLM for the autonomous Memanto backend.

Talks to any OpenAI-compatible endpoint (Ollama by default). Used to:
- write real prose answers in ``answer.generate``
- detect contradictions between memories (supersede / provenance)

Every call fails soft: if the LLM is unreachable or times out, callers
fall back to the extractive behaviour — the API never breaks.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class LocalLLMError(Exception):
    """Raised when the local LLM is unreachable or returns an error."""


class LocalLLMClient:
    """Thin OpenAI-compatible chat client (default: Ollama)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._chat_url = f"{self.base_url}/v1/chat/completions"

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Quick health probe — never raises."""
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Single-turn chat completion. Raises LocalLLMError on failure."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self._chat_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as e:
            raise LocalLLMError(f"Local LLM unavailable: {e}") from e

    # ------------------------------------------------------------------
    # Memanto-specific helpers
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        context: list[dict[str, Any]],
        header_prompt: str,
        footer_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Produce a prose answer from recalled memory context."""
        context_block = "\n\n".join(
            f"[{i + 1}] {doc.get('text', '')}" for i, doc in enumerate(context)
        )
        user = (
            f"Question: {query}\n\n"
            f"Memory context:\n{context_block}\n\n"
            f"{footer_prompt}"
        )
        return self.chat(
            system=header_prompt, user=user, temperature=temperature, max_tokens=512
        )

    def check_contradiction(
        self,
        new_memory: str,
        existing_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Ask the LLM whether ``new_memory`` contradicts any existing memory.

        Returns:
            {"contradicts": bool, "conflicting_ids": [...], "reason": str}
        """
        existing_block = "\n".join(
            f"- [{doc.get('id')}] {doc.get('text', '')}" for doc in existing_memories
        )
        system = (
            "You analyze agent memories. Determine whether the NEW memory "
            "contradicts any of the EXISTING memories. "
            'Reply with JSON only: {"contradicts": true|false, '
            '"conflicting_ids": [ids], "reason": "short explanation"}'
        )
        user = f"EXISTING MEMORIES:\n{existing_block}\n\nNEW MEMORY:\n{new_memory}"
        try:
            raw = self.chat(system=system, user=user, temperature=0.1, max_tokens=256)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return {
                "contradicts": bool(data.get("contradicts", False)),
                "conflicting_ids": list(data.get("conflicting_ids", [])),
                "reason": str(data.get("reason", "")),
            }
        except Exception:
            return {"contradicts": False, "conflicting_ids": [], "reason": ""}
