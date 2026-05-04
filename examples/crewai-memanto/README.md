# CrewAI + Memanto — Agentic Memory Integration

## 🎯 What This Does

Replaces standard CrewAI memory with **Memanto** — a typed semantic memory layer that persists across sessions and agents.

## 🏗️ Architecture

```
Research Agent                   Writer Agent
      │                               │
      │  store findings               │  retrieve findings
      ▼                               ▼
┌─────────────────────────────────────────┐
│           MEMANTO Memory Layer           │
│  • Typed memory (fact, preference, ...)  │
│  • Cross-session persistence            │
│  • Contradiction detection               │
│  • Semantic retrieval (no indexing)     │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install crewai memanto
```

### 2. Get a Moorcheh API key
- Sign up at https://console.moorcheh.ai/api-keys
- Free tier: 100K operations/month

### 3. Set your API key
```bash
export MOORCHEH_API_KEY="your-key-here"
```

### 4. Run the demo
```bash
python3 memory_test.py
```

## 📋 What the Demo Shows

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Research Agent stores 5 typed memories | ✅ |
| 2 | Simulated cross-session delay | ✅ |
| 3 | Writer Agent retrieves by semantic query | ✅ |
| 4 | Contradictory memory handling (bonus) | ✅ |

## 🔄 How to Swap Standard CrewAI Memory for Memanto

1. Replace `from crewai.memory import Memory` with `from memanto import MemantoClient`
2. Initialize: `memanto = MemantoClient(api_key=os.getenv("MOORCHEH_API_KEY"))`
3. Store: `memanto.remember(agent_name="my-agent", content="...", memory_type="fact")`
4. Retrieve: `memanto.recall(agent_name="my-agent", query="...", limit=5)`

## 📦 Submission

- **Issue:** https://github.com/moorcheh-ai/memanto/issues/37
- **Bounty:** $100 USD
- **Author:** Atlas Nexus (atlasnexus.ops@proton.me)
- **License:** MIT
