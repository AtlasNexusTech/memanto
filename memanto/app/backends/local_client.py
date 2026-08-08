"""
LocalMoorchehClient — drop-in replacement for ``moorcheh_sdk.MoorchehClient``.

Exposes the same surface used by the Memanto services:

- ``documents.upload / get / delete``
- ``namespaces.create / list / delete``
- ``similarity_search.query``
- ``answer.generate`` (extractive answer built from recalled context)

Everything runs on SQLite + the local embedding engine. No API key, no
network. Set ``MEMANTO_BACKEND=local`` to activate (default in the absence
of a Moorcheh API key).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from memanto.app.backends.embeddings import EmbeddingEngine
from memanto.app.backends.store import LocalStore


class _Documents:
    def __init__(self, store: LocalStore):
        self._store = store

    def upload(self, namespace_name: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        submitted: list[str] = []
        for doc in documents:
            doc_id = str(doc["id"])
            self._store.create_namespace(namespace_name)
            self._store.upsert_document(
                doc_id,
                namespace_name,
                str(doc.get("text", "")),
                doc.get("metadata"),
            )
            submitted.append(doc_id)
        return {"status": "success", "submitted_ids": submitted}

    def get(self, namespace_name: str, ids: list[str | int]) -> dict[str, Any]:
        return {"documents": self._store.get_documents(namespace_name, ids)}

    def delete(self, namespace_name: str, ids: list[str | int]) -> dict[str, Any]:
        deleted = self._store.delete_documents(namespace_name, ids)
        return {"status": "success", "deleted_ids": deleted}


class _Namespaces:
    def __init__(self, store: LocalStore):
        self._store = store

    def create(
        self, namespace_name: str, type: str, vector_dimension: int | None = None
    ) -> dict[str, Any]:
        ns = self._store.create_namespace(namespace_name, type, vector_dimension)
        return {
            "message": "Namespace created",
            "namespace_name": ns["namespace_name"],
            "type": ns["type"],
            "vector_dimension": ns["vector_dimension"],
        }

    def list(self) -> dict[str, Any]:
        start = time.time()
        namespaces = self._store.list_namespaces()
        return {"namespaces": namespaces, "execution_time": round(time.time() - start, 4)}

    def delete(self, namespace_name: str) -> None:
        self._store.delete_namespace(namespace_name)


class _SimilaritySearch:
    def __init__(self, store: LocalStore, engine: EmbeddingEngine):
        self._store = store
        self._engine = engine

    def query(
        self,
        namespaces: list[str],
        query: str | list[float],
        top_k: int = 10,
        threshold: float | None = None,
        kiosk_mode: bool = False,
    ) -> dict[str, Any]:
        start = time.time()
        if isinstance(query, list):
            # Vector query unsupported by the TF-IDF engine — fall back to
            # the first namespace's text search with the query serialized.
            query_text = " ".join(str(x) for x in query[:8])
        else:
            query_text = query

        results: list[dict[str, Any]] = []
        for ns in namespaces:
            docs = self._store.active_documents(ns)
            results.extend(
                self._engine.search(query_text, docs, top_k=top_k, threshold=threshold)
            )
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return {
            "results": results[:top_k],
            "execution_time": round(time.time() - start, 4),
        }


class _Answer:
    def __init__(self, store: LocalStore, engine: EmbeddingEngine):
        self._store = store
        self._engine = engine

    def generate(
        self,
        query: str,
        namespace: str | None = None,
        top_k: int | None = None,
        ai_model: str = "local-extractive",
        chat_history: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        header_prompt: str | None = None,
        footer_prompt: str | None = None,
        threshold: float | None = None,
        kiosk_mode: bool = False,
        structured_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        k = top_k or 5
        if namespace:
            namespaces = [namespace]
        else:
            namespaces = [ns["namespace_name"] for ns in self._store.list_namespaces()]

        search = _SimilaritySearch(self._store, self._engine).query(
            namespaces, query, top_k=k, threshold=threshold
        )
        results = search["results"]
        context_count = len(results)

        if context_count == 0:
            return {
                "answer": (
                    "Je n'ai trouvé aucun contexte pertinent dans la mémoire. "
                    "Aucune réponse générée."
                    if ai_model.startswith("local") and header_prompt is None
                    else "No relevant context found in memory."
                ),
                "model": ai_model,
                "context_count": 0,
                "query": query,
                "used_context": False,
                "structured_data": None,
            }

        # Extractive answer: rank the most relevant passages.
        passages = []
        for r in results[:3]:
            text = str(r.get("text", "")).strip()
            if text:
                passages.append(f"- {text}")
        answer = "\n".join(passages)
        if header_prompt:
            answer = f"{header_prompt}\n\n{answer}"
        if footer_prompt:
            answer = f"{answer}\n\n{footer_prompt}"

        return {
            "answer": answer,
            "model": ai_model,
            "context_count": context_count,
            "query": query,
            "used_context": True,
            "structured_data": structured_response,
        }


class LocalMoorchehClient:
    """Moorcheh-compatible client backed by SQLite + local embeddings."""

    def __init__(self, db_path: str | Path = "memanto.db", use_fastembed: bool = False):
        self.api_key = "local"
        self.base_url = "local://sqlite"
        self.timeout = 0
        self._store = LocalStore(db_path)
        self._engine = EmbeddingEngine(use_fastembed=use_fastembed)
        self.documents = _Documents(self._store)
        self.namespaces = _Namespaces(self._store)
        self.similarity_search = _SimilaritySearch(self._store, self._engine)
        self.answer = _Answer(self._store, self._engine)

    def close(self) -> None:
        self._store.close()
