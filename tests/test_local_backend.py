"""
Tests for the local (autonomous) Memanto backend.

Run without any API key:  pytest tests/test_local_backend.py
"""

import os
import tempfile

import pytest

from memanto.app.backends.local_client import LocalMoorchehClient

MEMORIES = [
    {
        "id": "m1",
        "text": "[FACT] Contact client\n\nLe client Acme prefere etre contacte par email le matin.",
        "metadata": {"memory_type": "fact", "confidence": 0.9},
    },
    {
        "id": "m2",
        "text": "[PREFERENCE] Budget\n\nLe budget de l'offre est de 5000 euros.",
        "metadata": {"memory_type": "preference"},
    },
    {
        "id": "m3",
        "text": "[FACT] Contrat\n\nLe contrat actuel expire en juin.",
        "metadata": {"memory_type": "fact"},
    },
]


@pytest.fixture()
def client():
    db = tempfile.mktemp(suffix=".db")
    c = LocalMoorchehClient(db_path=db)
    yield c
    c.close()
    if os.path.exists(db):
        os.unlink(db)


def test_namespace_lifecycle(client):
    resp = client.namespaces.create("agent-test", "agent")
    assert resp["namespace_name"] == "agent-test"
    assert resp["type"] == "agent"

    listed = client.namespaces.list()
    names = [ns["namespace_name"] for ns in listed["namespaces"]]
    assert "agent-test" in names

    client.namespaces.delete("agent-test")
    listed = client.namespaces.list()
    names = [ns["namespace_name"] for ns in listed["namespaces"]]
    assert "agent-test" not in names


def test_upload_and_get(client):
    client.namespaces.create("agent-test", "agent")
    resp = client.documents.upload("agent-test", MEMORIES)
    assert resp["status"] == "success"
    assert set(resp["submitted_ids"]) == {"m1", "m2", "m3"}

    got = client.documents.get("agent-test", ["m1", "m2"])["documents"]
    assert {d["id"] for d in got} == {"m1", "m2"}
    assert got[0]["metadata"]["memory_type"] == "fact"


def test_upload_replaces_existing_document(client):
    client.namespaces.create("agent-test", "agent")
    client.documents.upload("agent-test", [MEMORIES[0]])
    updated = dict(MEMORIES[0])
    updated["text"] = "[FACT] Contact client\n\nLe client Acme prefere un appel en fin de journee."
    client.documents.upload("agent-test", [updated])

    got = client.documents.get("agent-test", ["m1"])["documents"]
    assert "appel en fin de journee" in got[0]["text"]


def test_semantic_recall_ranks_relevant_memory_first(client):
    client.namespaces.create("agent-test", "agent")
    client.documents.upload("agent-test", MEMORIES)

    res = client.similarity_search.query(
        ["agent-test"], "comment contacter ce client", top_k=3
    )["results"]
    assert res[0]["id"] == "m1"
    assert res[0]["score"] > 0.1

    res = client.similarity_search.query(
        ["agent-test"], "quand expire le contrat", top_k=3
    )["results"]
    assert res[0]["id"] == "m3"


def test_threshold_filters_weak_matches(client):
    client.namespaces.create("agent-test", "agent")
    client.documents.upload("agent-test", MEMORIES)

    res = client.similarity_search.query(
        ["agent-test"], "xyzzy inconnu", top_k=5, threshold=0.05
    )["results"]
    assert res == []


def test_answer_generates_from_context(client):
    client.namespaces.create("agent-test", "agent")
    client.documents.upload("agent-test", MEMORIES)

    ans = client.answer.generate("quel est le budget", namespace="agent-test")
    assert ans["context_count"] >= 1
    assert ans["used_context"] is True
    assert "5000" in ans["answer"]

    ans_empty = client.answer.generate(
        "xyzzy inconnu", namespace="agent-test", threshold=0.05
    )
    assert ans_empty["context_count"] == 0
    assert ans_empty["used_context"] is False


def test_delete_soft_removes_from_recall(client):
    client.namespaces.create("agent-test", "agent")
    client.documents.upload("agent-test", MEMORIES)

    resp = client.documents.delete("agent-test", ["m2"])
    assert resp["deleted_ids"] == ["m2"]

    res = client.similarity_search.query(["agent-test"], "budget", top_k=5)["results"]
    assert "m2" not in [r["id"] for r in res]

    # History preserved: document no longer returned by get
    assert client.documents.get("agent-test", ["m2"])["documents"] == []
