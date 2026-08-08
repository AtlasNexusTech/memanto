# Memanto — Mode local autonome (SQLite)

Memanto fonctionne maintenant **sans compte cloud, sans clé API** grâce au
backend local : vos agents gardent une mémoire persistante dans un fichier
SQLite, avec recherche sémantique par embeddings.

```
┌────────────────────┐     ┌──────────────────────────────────────┐
│  Agent / CLI / API │ ──▶ │  Memanto (FastAPI)                   │
└────────────────────┘     │  MEMANTO_BACKEND=local               │
                           │  ├─ SQLite (memanto.db)              │
                           │  │   namespaces / documents          │
                           │  └─ EmbeddingEngine (TF-IDF + stems) │
                           └──────────────────────────────────────┘
```

## Activation

```bash
# 1. Aucune clé API requise. Choisir le backend local :
export MEMANTO_BACKEND=local
export MEMANTO_DB_PATH=./memanto.db

# 2. Lancer l'API
uvicorn memanto.app.main:app --port 8000
```

| Variable | Valeur | Défaut |
|---|---|---|
| `MEMANTO_BACKEND` | `auto` \| `local` \| `moorcheh` | `auto` |
| `MEMANTO_DB_PATH` | chemin du fichier SQLite | `memanto.db` |

- **`auto`** : backend local si `MOORCHEH_API_KEY` est vide, cloud sinon.
- **`local`** : force le backend local (aucune clé nécessaire).
- **`moorcheh`** : force le cloud Moorcheh (clé requise).

En mode local, n'importe quel Bearer token est accepté (`Authorization: Bearer local`).

## Démo rapide (HTTP)

```bash
AUTH="Authorization: Bearer local"

# Créer un agent + activer une session
curl -X POST localhost:8000/api/v2/agents -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"agent-hermes","display_name":"Hermes"}'

SESSION=$(curl -X POST localhost:8000/api/v2/agents/agent-hermes/activate \
  -H "$AUTH" | python3 -c "import json,sys;print(json.load(sys.stdin)['session_token'])")

# Mémoriser
curl -X POST localhost:8000/api/v2/agents/agent-hermes/remember \
  -H "$AUTH" -H "X-Session-Token: $SESSION" -H "Content-Type: application/json" \
  -d '{"memory_type":"fact","title":"Contact client",
       "content":"Le client Acme prefere etre contacte par email le matin.","confidence":0.9}'

# Rappel sémantique
curl -X GET "localhost:8000/api/v2/agents/agent-hermes/recall?query=comment%20contacter%20ce%20client" \
  -H "$AUTH" -H "X-Session-Token: $SESSION"
```

## Interface répliquée

Le client local expose la même surface que `moorcheh_sdk.MoorchehClient` :

- `documents.upload / get / delete`
- `namespaces.create / list / delete`
- `similarity_search.query(namespaces, query, top_k, threshold)`
- `answer.generate(...)` — réponse extractive construite à partir du
  contexte rappelé (pas de LLM externe requis)

## Recherche sémantique

`EmbeddingEngine` (dans `memanto/app/backends/embeddings.py`) :
- Vectorisation **TF-IDF** pure Python + numpy (aucun téléchargement de modèle)
- Stemming léger FR/EN (contacte/contacter → même racine)
- Similarité cosinus, seuil optionnel (`threshold`)
- Optionnel : si `fastembed` est installé, il est utilisé pour de vraies
  embeddings (dimension 384)

## Stockage

`LocalStore` (dans `memanto/app/backends/store.py`) — SQLite avec :
- Isolation par namespace (chaque agent a son namespace `memanto_agent_<id>`)
- Métadonnées plates (JSON) pour filtrage futur (`#memory_type:fact`)
- Suppression douce (`active=0`) : l'historique est conservé pour la
  provenance et le supersede
- `PRAGMA journal_mode=WAL` pour lecture/écriture concurrentes

## Tests

```bash
python -m pytest tests/test_local_backend.py -q
```

Couvre : cycle de vie des namespaces, upload/get, remplacement par id,
recall sémantique (rang du document pertinent en premier), filtre par seuil,
réponse extractive, suppression douce.

## Limitations actuelles

- **Détection de contradiction / supersede** : le stockage et le rappel
  fonctionnent en local, mais la détection automatique des contradictions
  (qui reposait sur un LLM cloud) n'est pas encore portée en local.
- **`answer.generate`** : réponse extractive (meilleurs passages) — pas une
  génération LLM. La génération d'un résumé rédigé nécessite un LLM local
  ou le mode cloud.
- **Résumés quotidiens / conflits** : nécessitent un LLM (mode cloud ou
  branchement ultérieur sur un LLM local).
