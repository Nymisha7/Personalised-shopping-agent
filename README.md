This is my submission to the OPENENV hackathon

Link:https://huggingface.co/spaces/Nymisha7/RL-Personality-automatic-shopper

---
title: Shopping Agent OpenEnv
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - fastapi
license: bsd-3-clause
---

# 🧠 Personality-Driven Shopping Agent — OpenEnv Environment

A **real-world shopping simulation** where an AI agent must learn a specific user's shopping personality from persistent memory and make purchase decisions aligned with their style — not just pick the "globally best" product.

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification for reinforcement learning environments.

## 🎯 Motivation

Current shopping recommenders optimize generic objectives (clicks, conversions, popularity). They don't capture an **individual user's actual decision process**.

This environment challenges agents to answer:
> "Given multiple products, which one would **this specific user** most likely choose, and why?"

The agent must reason over **10 personality dimensions** loaded from persistent memory files, making it a genuine real-world task that humans do every day.

---

## 🏗️ Architecture

```
User behavior
   ↓
memory/ (personality.md, preferences.json, semantic_memory.jsonl)
   ↓
product_generator → Dynamic product catalog with decoys/traps
   ↓
shopping_env.py → OpenEnv-compliant RL environment
   ↓
personality_grader.py → Deterministic reward function (0.0–1.0)
   ↓
inference.py → LLM agent using OpenAI client
```

---

## 📋 Tasks (Easy → Medium → Hard)

| Task | Name | Query | Products | Max Steps | Difficulty |
|------|------|-------|----------|-----------|------------|
| 🟢 Easy | `quick_pick` | Lip balm | 4 | 8 | Minimal traps, basic reasoning |
| 🟡 Medium | `smart_shop` | Earbuds | 8 | 12 | Budget traps, requires research |
| 🔴 Hard | `expert_deal` | Laptop backpack | 12 | 15 | Multiple decoys, close competitors |

### Grader Criteria (deterministic, scores 0.0–1.0)

Each task is scored using `personality_grader.grade_purchase()`:

- **Base score (0–0.60)**: Personality alignment of the purchased product vs. the ideal product
- **Research bonus (0–0.25)**: Viewing, comparing, and shortlisting products (scaled by user's `research_depth` trait)
- **Risk avoidance bonus (0–0.10)**: Skipping suspicious/cheap products (scaled by `risk_aversion`)
- **Penalties (-0.05 to -0.10)**: Buying multiple items, buying risky items, running out of steps

---

## 🔧 Action Space (`ShoppingAction`)

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `search` | Search the catalog | `search_query` |
| `view_item` | View product details | `item_ids` |
| `compare` | Compare 2+ products | `item_ids` (≥2) |
| `shortlist` | Mark products for later | `item_ids` |
| `add_to_cart` | Add to cart | `item_ids` |
| `remove_from_cart` | Remove from cart | `item_ids` |
| `buy` | Purchase items in cart + item_ids | `item_ids` (optional) |
| `skip` | Skip turn / skip products | `item_ids` (optional) |
| `ask_more` | Request more options | — |

```json
{
  "action_type": "view_item",
  "item_ids": ["p1", "p3"],
  "search_query": null
}
```

---

## 👁️ Observation Space (`ShoppingObservation`)

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Current search query |
| `category` | string | Product category |
| `candidate_products` | list[dict] | Products with id, name, price, rating, brand, reviews, seller, refundable, features |
| `memory_profile` | dict | User's personality traits, goals, semantic conclusions |
| `cart` | list[str] | Product IDs in cart |
| `shortlisted` | list[str] | Shortlisted product IDs |
| `viewed_items` | list[str] | Viewed product IDs |
| `compared_sets` | list[list[str]] | Comparison groups |
| `history_summary` | string | Recent action history |
| `feedback` | string | Environment feedback on last action |
| `step_number` | int | Current step |
| `max_steps` | int | Maximum steps in episode |

---

## 📊 Reward Function

Multi-signal reward providing feedback **over the full trajectory** (not just binary end-of-episode):

| Signal | Reward | Condition |
|--------|--------|-----------|
| View items | +0.05 × research_depth | Per item viewed |
| Compare items | +0.05 × count × research_depth | Per comparison |
| Shortlist | +0.10 × research_depth | When shortlisting |
| Add to cart | +0.10 | Per item added |
| **Buy (grader)** | **0.0 – 0.85** | **Personality alignment score** |
| Invalid action | -0.10 | Unknown action type |
| Out of steps | -0.20 | Timeout without purchase |

---

## 🧬 Personality Dimensions (from `memory/`)

The user profile is loaded from persistent memory files:

| Dimension | What It Controls |
|-----------|------------------|
| `price_sensitivity` | Prefers cheaper vs. premium products |
| `quality_preference` | Weight given to ratings |
| `risk_aversion` | Avoids unknown sellers, low-review products |
| `research_depth` | How much viewing/comparing before buying |
| `brand_trust` | Preference for known brands |
| `review_dependence` | Importance of review count |
| `return_preference` | Preference for refundable products |
| `discount_sensitivity` | Attraction to sales/discounts |
| `exploration_vs_repeat` | Tries new products vs. sticks with known |
| `decision_speed` | Quick decisions vs. deliberation |

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Open web UI
# http://localhost:7860
```

### Run Inference (Baseline Scores)

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export MODEL_NAME="gpt-4.1-mini"
export BASELINE_RANDOM_SEED="42"

# Run all 3 tasks
python inference.py
```

Optional compatibility mode:

```bash
# If you are using a non-OpenAI compatible provider, you can still override:
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-hf-token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

### Docker

```bash
# Build
docker build -t shopping-agent-env:latest .

# Run
docker run -p 7860:7860 shopping-agent-env:latest

# Test health
curl http://localhost:7860/health
```

Verified locally:
- `docker build -t shopping-agent-env:local .`
- `docker run -p 7860:7860 shopping-agent-env:local`
- `GET /health` returns `{"status":"healthy"}`

### Hugging Face Space Deployment

This repository is ready to run as a Docker-based Hugging Face Space and is tagged for OpenEnv in the README frontmatter above.

```bash
# Create a Docker Space, then push this repo to it
git remote add hf https://huggingface.co/spaces/<username>/shopping-agent-openenv
git push hf main
```

Space settings:
- SDK: `Docker`
- App port: `7860`
- Tag: `openenv`

Deployment checklist:
- Create the Space with SDK set to `Docker`
- Push this repository as-is
- Confirm the README frontmatter remains at the top of the repo README
- Verify `/health` after the Space finishes building

After deployment, verify:

```bash
curl https://<username>-shopping-agent-openenv.hf.space/health
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (accepts `task` or `query`) |
| `/step` | POST | Execute action |
| `/state` | GET | Get current episode state |
| `/` | GET | Interactive web UI |
| `/profile` | GET | View user personality profile |

### Reset with a task:
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "quick_pick"}'
```

### Step with an action:
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "view_item", "item_ids": ["p1", "p2"]}'
```

---

## 📈 Baseline Scores

Using heuristic fallback agent (no LLM):

| Task | Difficulty | Score | Steps |
|------|------------|-------|-------|
| `quick_pick` | Easy | ~0.65 | 5-6 |
| `smart_shop` | Medium | ~0.55 | 8-10 |
| `expert_deal` | Hard | ~0.40 | 12-15 |

With Qwen2.5-72B-Instruct:

| Task | Difficulty | Score | Steps |
|------|------------|-------|-------|
| `quick_pick` | Easy | ~0.80 | 4-5 |
| `smart_shop` | Medium | ~0.70 | 7-9 |
| `expert_deal` | Hard | ~0.55 | 10-13 |

---

## 📁 Project Structure

```
├── openenv.yaml           # Environment manifest (3 tasks)
├── models.py              # Pydantic models (Action, Observation, State, StepResult)
├── shopping_env.py        # OpenEnv-compliant environment
├── personality_grader.py  # Deterministic grader (0.0–1.0)
├── product_generator.py   # Dynamic product catalog generator
├── memory_engine.py       # Memory loader (personality, preferences)
├── rl_agent.py            # Autonomous RL agent (weight-based policy)
├── inference.py           # Baseline inference script
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── server/
│   ├── app.py             # FastAPI server
│   └── __init__.py
├── static/
│   └── index.html         # Interactive web UI
└── memory/
    ├── personality.md      # User shopping personality
    ├── preferences.json    # Structured trait values
    ├── semantic_memory.jsonl # Stable learned conclusions
    ├── episodic_log.jsonl  # Shopping event history
    └── projects.md         # Active shopping projects
```

---

## 🔬 What Makes This Unique

1. **Real-world task**: People actually shop this way — comparing options, checking reviews, weighing tradeoffs
2. **Personality-conditioned**: Not just "find the best product" but "find what THIS USER would pick"
3. **10-dimensional scoring**: Multi-factor grading captures nuanced personality alignment
4. **Dynamic catalogs**: Products are generated per-episode, preventing memorization
5. **Trap products**: Suspiciously cheap items, unknown sellers, and close competitors test genuine reasoning
6. **Partial-progress rewards**: Research actions earn rewards, not just the final purchase
7. **Persistent memory**: Agent can learn from past episodes via episodic and semantic memory

---

## 📜 License

BSD-3-Clause
