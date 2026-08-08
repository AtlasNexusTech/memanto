"""
Embedding engine for the local Memanto backend.

Pure-Python + numpy TF-IDF vectorization with cosine similarity — no model
download, no GPU, no API key. Good enough for semantic recall on typical
agent memory workloads, and fully deterministic.

If ``fastembed`` is installed it is used as an optional semantic upgrade
(real embeddings); otherwise the TF-IDF fallback keeps everything local.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", re.IGNORECASE)

# Light FR/EN suffix stripping to improve lexical matching between
# inflected forms (contacte/contacter, budget/budgets, running/run).
_SUFFIXES = (
    "ateur",
    "ation",
    "ements",
    "ement",
    "ateurs",
    "atrice",
    "issaient",
    "issant",
    "issait",
    "isses",
    "isse",
    "irais",
    "irait",
    "issez",
    "issa",
    "erais",
    "aient",
    "ions",
    "erez",
    "eras",
    "erai",
    "ant",
    "ais",
    "ait",
    "ées",
    "ée",
    "és",
    "es",
    "é",
    "s",
    "ent",
    "er",
    "e",
    "x",
)

try:  # optional semantic upgrade
    from fastembed import TextEmbedding  # type: ignore[import-not-found]

    _HAVE_FASTEMBED = True
except Exception:  # pragma: no cover
    _HAVE_FASTEMBED = False


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _WORD_RE.findall(text.lower())]


def _stem(word: str) -> str:
    """Aggressive but tiny stemmer: strip the longest known suffix once."""
    if len(word) <= 4:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


class EmbeddingEngine:
    """Stateless TF-IDF engine with optional fastembed backend."""

    def __init__(self, use_fastembed: bool = False):
        self.use_fastembed = use_fastembed and _HAVE_FASTEMBED
        self._model: Any = None
        if self.use_fastembed:
            self._model = TextEmbedding()

    @property
    def vector_dimension(self) -> int | None:
        if self._model is not None:
            return 384  # default fastembed model dimension
        return None  # TF-IDF: dynamic vocabulary

    # ------------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------------
    def _idf(self, corpus: list[str]) -> dict[str, float]:
        n = len(corpus)
        if n == 0:
            return {}
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(_tokenize(doc)))
        return {
            term: math.log((1 + n) / (1 + count)) + 1.0
            for term, count in df.items()
        }

    def vectorize(
        self, texts: list[str]
    ) -> tuple[list[dict[str, float]], dict[str, float]]:
        """Return (tf-idf sparse vectors as dicts, idf table)."""
        idf = self._idf(texts)
        vectors: list[dict[str, float]] = []
        for text in texts:
            tf = Counter(_tokenize(text))
            total = sum(tf.values()) or 1
            vec: dict[str, float] = {}
            for term, count in tf.items():
                if term in idf:
                    vec[term] = (count / total) * idf[term]
            vectors.append(vec)
        return vectors, idf

    def _norm(self, vec: dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) or 1.0

    def cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        return dot / (self._norm(a) * self._norm(b))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rank documents by cosine similarity to the query."""
        if not documents:
            return []
        if self._model is not None:  # pragma: no cover — fastembed path
            return self._search_fastembed(query, documents, top_k, threshold)

        docs = [d["text"] for d in documents]
        vectors, _ = self.vectorize(docs + [query])
        qvec = vectors[-1]
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc, vec in zip(documents, vectors[:-1]):
            score = self.cosine(qvec, vec)
            if threshold is None or score >= threshold:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**doc, "score": round(score, 4)}
            for score, doc in scored[:top_k]
        ]

    def _search_fastembed(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
        threshold: float | None,
    ) -> list[dict[str, Any]]:  # pragma: no cover — optional path
        import numpy as np  # type: ignore[import-not-found]

        model = self._model
        q = next(iter(model.embed([query]))).astype(np.float32)
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in documents:
            dvec = next(iter(model.embed([doc["text"]]))).astype(np.float32)
            score = float(np.dot(q, dvec) / (np.linalg.norm(q) * np.linalg.norm(dvec)))
            if threshold is None or score >= threshold:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**doc, "score": round(score, 4)} for score, doc in scored[:top_k]]
