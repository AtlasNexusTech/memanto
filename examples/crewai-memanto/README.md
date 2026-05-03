# CrewAI + Memanto: Persistent Agentic Memory

> **Replace CrewAI's volatile memory with Memanto's semantic, cross-session memory layer.**

This example demonstrates how to integrate [Memanto](https://memanto.ai) as the primary memory backend for [CrewAI](https://crewai.com) multi-agent workflows — enabling agents to **remember across sessions, recall past research, and handle contradictory memories**.

---

## 🎯 What This Solves

CrewAI agents are powerful but suffer from **long-term amnesia**: when a session ends, context is lost. Memanto provides a **persistent, searchable, semantic memory** that survives sessions.

| Without Memanto | With Memanto |
|----------------|--------------|
| ❌ Memory resets every run | ✅ Memory persists across days |
| ❌ Agents repeat research | ✅ Agents recall past findings |
| ❌ No contradiction handling | ✅ Old facts get superseded |
| ❌ Short-term only | ✅ Long-term semantic memory |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  CrewAI Crew                      │
│  ┌──────────────┐         ┌──────────────┐       │
│  │Research Agent│────────▶│ Writer Agent  │       │
│  │  (stores)    │         │  (recalls)    │       │
│  └──────┬───────┘         └──────┬────────┘       │
│         │                        │                │
│         ▼                        ▼                │
│  ┌──────────────────────────────────────────┐    │
│  │       MemantoCrewMemory (Adapter)         │    │
│  │  store() • recall() • update_fact()      │    │
│  └────────────────────┬─────────────────────┘    │
│                       │                           │
└───────────────────────┼───────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│               Memanto Memory Layer                 │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐   │
│  │ Semantic│  │ Namespace │  │Contradiction   │   │
│  │ Search  │  │ Isolation │  │ Resolution     │   │
│  └─────────┘  └──────────┘  └────────────────┘   │
│              Powered by Moorcheh                   │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Moorcheh API Key](https://console.moorcheh.ai/api-keys) (free tier available)

### Installation

```bash
# Install both packages
pip install crewai memanto

# Or with pipx
pipx install crewai
pipx install memanto
```

### Run the Memory Test

```bash
# Set your API key
export MOORCHEH_API_KEY='your-key-here'

# Run the demo
python examples/crewai-memanto/crew_with_memanto.py
```

---

## 🧪 The Memory Test (Bounty Requirement)

The demo script runs through 5 phases:

### Phase 1: Research → Store
Research Agent investigates "AI Agent Memory Systems" and stores findings in Memanto.

### Phase 2: Verify Persistence
Confirm memories are stored and retrievable.

### Phase 3: Recall (24h Later)
Writer Agent, in a **new session** (fresh client connection), queries Memanto and retrieves the Research Agent's findings from 24 hours ago.

### Phase 4: Report from Memory
Writer Agent produces a report **using only Memanto-recalled memories** — no fresh research.

### Phase 5 (Bonus): Contradictory Memories
```
Stored:  "User prefers dark mode"     (fact_id: abc123)
Updated: "User prefers light mode"    (supersedes: abc123)
Result:  Only the latest fact is active ✓
```

Old facts get marked as `superseded` with a provenance chain, so agents always get the most current information.

---

## 📦 How to Swap Standard CrewAI Memory for Memanto

```python
# BEFORE: Standard CrewAI (volatile memory)
from crewai import Crew

crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    # memory is lost when process exits
)

# AFTER: Memanto-powered CrewAI (persistent memory)
from examples.crewai_memanto.crew_with_memanto import MemantoCrewMemory

crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=MemantoCrewMemory(api_key="your-key", agent_id="my-crew"),
)
```

That's it. All agent memory operations now route through Memanto.

---

## 🔧 API Reference

### `MemantoCrewMemory`

| Method | Description |
|--------|-------------|
| `store(key, value, agent)` | Store a memory fact |
| `recall(query, limit=5)` | Semantic search of memories |
| `update_fact(id, new_content)` | Update + supersede old fact |
| `get_history(agent, limit)` | List memories for an agent |
| `reset()` | Clear all memories (testing) |

### MemoryRecord fields for filtering

Use Moorcheh's `#` syntax:
```
#memory_type:fact  → only fact memories
#actor_id:research_agent  → only research agent's memories
#confidence>0.8  → only high-confidence memories
```

---

## 🎥 Visual Proof

Run with Asciinema recording:

```bash
# Record the demo
asciinema rec memanto-crewai-demo.cast

# Run the memory test
python examples/crewai-memanto/crew_with_memanto.py

# Stop recording (Ctrl+D)
```

Or generate a high-quality GIF:

```bash
# Using terminalizer or similar
terminalizer record demo -c config.yml
```

---

## 📁 File Structure

```
examples/crewai-memanto/
├── crew_with_memanto.py   # Main integration + 5-phase memory test
├── README.md              # This file
└── requirements.txt       # crewai, memanto
```

---

## 🤝 Contributing

This is a bounty submission for [Issue #37](https://github.com/moorcheh-ai/memanto/issues/37).

- **Author**: Atlas Nexus Ops
- **Bounty**: $100 — Best-in-Class Integration: CrewAI + Memanto
- **Status**: Submitted

---

**Built with ❤️ by Atlas Nexus** — AI-powered operations.
