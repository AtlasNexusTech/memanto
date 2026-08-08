"""
Local SQLite store for the autonomous Memanto backend.

Schema keeps the full document lifecycle: namespaces, documents with flat
metadata, soft-delete (active flag) and timestamps. Search is delegated to
the EmbeddingEngine (TF-IDF over the active documents of a namespace).

No external service, no API key: everything lives in one SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class LocalStore:
    """SQLite-backed document store with namespace isolation."""

    def __init__(self, db_path: str | Path = "memanto.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS namespaces (
                    namespace_name TEXT PRIMARY KEY,
                    type TEXT NOT NULL DEFAULT 'general',
                    vector_dimension INTEGER,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    namespace_name TEXT NOT NULL REFERENCES namespaces(namespace_name) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_documents_namespace
                    ON documents(namespace_name);
                CREATE INDEX IF NOT EXISTS idx_documents_active
                    ON documents(namespace_name, active);
                """
            )

    # ------------------------------------------------------------------
    # Namespaces
    # ------------------------------------------------------------------
    def create_namespace(
        self, namespace_name: str, type_: str = "general", vector_dimension: int | None = None
    ) -> dict[str, Any]:
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO namespaces (namespace_name, type, vector_dimension, created_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace_name, type_, vector_dimension, now),
            )
        ns = self.get_namespace(namespace_name)
        assert ns is not None
        return ns

    def get_namespace(self, namespace_name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT namespace_name, type, vector_dimension, created_at FROM namespaces "
            "WHERE namespace_name = ?",
            (namespace_name,),
        ).fetchone()
        if row is None:
            return None
        count = self._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE namespace_name = ? AND active = 1",
            (namespace_name,),
        ).fetchone()[0]
        return {
            "namespace_name": row["namespace_name"],
            "type": row["type"],
            "item_count": count,
            "vector_dimension": row["vector_dimension"],
        }

    def list_namespaces(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT namespace_name FROM namespaces").fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            ns = self.get_namespace(r["namespace_name"])
            if ns is not None:
                out.append(ns)
        return out

    def delete_namespace(self, namespace_name: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM namespaces WHERE namespace_name = ?", (namespace_name,)
            )

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    def upsert_document(
        self,
        doc_id: str,
        namespace_name: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._conn:
            self._conn.execute(
                "INSERT INTO documents (id, namespace_name, text, metadata_json, created_at, updated_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(id) DO UPDATE SET text = excluded.text, "
                "metadata_json = excluded.metadata_json, updated_at = excluded.updated_at, active = 1",
                (doc_id, namespace_name, text, meta_json, now, now),
            )

    def get_documents(self, namespace_name: str, ids: list[str | int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, text, metadata_json FROM documents "
            f"WHERE namespace_name = ? AND id IN ({placeholders}) AND active = 1",
            [namespace_name, *[str(i) for i in ids]],
        ).fetchall()
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]

    def active_documents(self, namespace_name: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, text, metadata_json FROM documents "
            "WHERE namespace_name = ? AND active = 1 ORDER BY created_at",
            (namespace_name,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]

    def delete_documents(self, namespace_name: str, ids: list[str | int]) -> list[str]:
        """Soft-delete (keeps history for provenance / supersede)."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        str_ids = [str(i) for i in ids]
        with self._conn:
            self._conn.execute(
                f"UPDATE documents SET active = 0, updated_at = ? "
                f"WHERE namespace_name = ? AND id IN ({placeholders})",
                [time.time(), namespace_name, *str_ids],
            )
        return str_ids

    def close(self) -> None:
        self._conn.close()
