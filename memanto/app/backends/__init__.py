"""Local backend for autonomous Memanto (SQLite + TF-IDF embeddings)."""

from memanto.app.backends.embeddings import EmbeddingEngine
from memanto.app.backends.llm_client import LocalLLMClient
from memanto.app.backends.local_client import LocalMoorchehClient
from memanto.app.backends.store import LocalStore

__all__ = ["EmbeddingEngine", "LocalLLMClient", "LocalMoorchehClient", "LocalStore"]
