"""
FastAPI server for the Personality-Driven Shopping Agent Environment.

OpenEnv-compliant HTTP API:
  POST /reset         → Reset environment, returns StepResult
  POST /step          → Execute an action, returns StepResult
  GET  /state         → Returns current ShoppingState
  GET  /health        → {"status": "healthy"}

Also serves:
  GET  /              → Interactive web UI
  POST /auto-run      → Autonomous RL episode (for web UI)
  GET  /agent-stats   → Agent learning stats
  POST /agent-reset   → Reset agent weights
  GET  /profile       → User personality profile
"""

import os
import json
import yaml
import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure parent directory is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ResetRequest, StepResult, ShoppingAction, ShoppingState
from shopping_env import ShoppingEnv
from memory_engine import load_profile
from personality_grader import score_all_products
from rl_agent import RLShoppingAgent

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
YAML_PATH = Path(__file__).resolve().parent.parent / "openenv.yaml"

# Load task configs from openenv.yaml
TASK_CONFIGS = {}
if YAML_PATH.exists():
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        _yaml = yaml.safe_load(f)
        for task in _yaml.get("tasks", []):
            TASK_CONFIGS[task["name"]] = task


# ---------------------------------------------------------------------------
# RL Agent — persists across requests
# ---------------------------------------------------------------------------
_agent = RLShoppingAgent()
_env: ShoppingEnv = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = ShoppingEnv()
    yield
    await _env.close()


app = FastAPI(
    title="Personality-Driven Shopping Agent",
    description=(
        "OpenEnv-compliant RL environment. The AI agent learns to shop "
        "like YOU by reading your memory/ profile. Supports 3 tasks: "
        "quick_pick (easy), smart_shop (medium), expert_deal (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse(content={"error": "Frontend not found"}, status_code=404)


# ---------------------------------------------------------------------------
# OpenEnv Core Endpoints: /reset, /step, /state, /health
# ---------------------------------------------------------------------------

class ResetBody(BaseModel):
    """Body for /reset — accepts task name or free-form query."""
    task: Optional[str] = Field(
        default=None,
        description="Task name: quick_pick, smart_shop, or expert_deal",
    )
    query: Optional[str] = Field(
        default=None,
        description="Free-form product query (used if task not specified)",
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(body: ResetBody = ResetBody()):
    """
    Reset the environment. OpenEnv-compliant.
    
    Accepts either:
      - task: "quick_pick" | "smart_shop" | "expert_deal"
      - query: free-form product query
    
    Returns StepResult with initial observation.
    """
    global _env

    # Resolve task config
    task_name = body.task or "smart_shop"
    task_config = TASK_CONFIGS.get(task_name, {})
    query = body.query or task_config.get("query", "earbuds")
    max_steps = task_config.get("max_steps", 12)
    product_count = task_config.get("product_count", 8)
    difficulty = task_config.get("difficulty", "medium")

    _env = ShoppingEnv(task_name=task_name)
    _env._max_steps = max_steps
    result = await _env.reset(query=query, product_count=product_count)

    # Add task metadata to info
    profile = _env._user_profile
    prefs = profile.get_prefs_for_category(query.lower())
    scored = _env._scored_products

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": {
            "task": task_name,
            "difficulty": difficulty,
            "query": query,
            "product_count": len(_env.catalog),
            "max_steps": max_steps,
            "personality_traits": prefs,
            "ideal_product": scored[0]["product"]["name"] if scored else None,
            "ideal_score": scored[0]["personality_score"] if scored else 0,
            "scored_products": [
                {
                    "id": s["product"]["id"],
                    "name": s["product"]["name"],
                    "score": s["personality_score"],
                    "rank": s["rank"],
                }
                for s in scored
            ],
        },
    }


@app.post("/step")
async def step(action: ShoppingAction):
    """
    Execute an action. OpenEnv-compliant.
    
    Returns StepResult with observation, reward, done, info.
    """
    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized. Call /reset first."},
        )
    result = await _env.step(action)
    state = await _env.state()
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": {
            "step_count": state.step_count,
            "cumulative_reward": state.cumulative_reward,
            "cart": state.cart,
        },
    }


@app.get("/state")
async def state():
    """
    Get current episode state. OpenEnv-compliant.
    """
    if _env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized."},
        )
    s = await _env.state()
    return s.model_dump()


# ---------------------------------------------------------------------------
# Autonomous RL Endpoints (for web UI)
# ---------------------------------------------------------------------------

@app.post("/auto-run")
async def auto_run():
    """
    Run a FULL autonomous RL episode.
    No human feedback needed — personality_grader IS the reward function.
    """
    if _env is None or not _env.catalog:
        return JSONResponse(
            status_code=400,
            content={"error": "No active session. Call /reset first."},
        )

    attempts = _agent.run_episode(
        products=_env.catalog,
        scored_products=_env._scored_products,
    )

    ideal = _env._scored_products[0] if _env._scored_products else None
    stats = _agent.get_stats()

    return {
        "attempts": attempts,
        "episode_number": stats["episode_count"],
        "total_episode_reward": round(
            sum(a["reward"] for a in attempts), 4
        ),
        "success": any(a["is_success"] for a in attempts),
        "success_attempt": next(
            (a["attempt"] for a in attempts if a["is_success"]), None
        ),
        "ideal_product": {
            "name": ideal["product"]["name"],
            "score": ideal["personality_score"],
            "id": ideal["product"]["id"],
        } if ideal else None,
        "agent_stats": stats,
    }


@app.get("/agent-stats")
async def agent_stats():
    """Get the agent's learning statistics and current weights."""
    return _agent.get_stats()


@app.post("/agent-reset")
async def agent_reset():
    """Reset the agent's learned weights — start fresh."""
    _agent.reset_weights()
    return {
        "message": "Agent weights reset. Learning starts from scratch.",
        "stats": _agent.get_stats(),
    }


@app.get("/profile")
async def profile():
    """Returns the loaded user personality profile summary."""
    prof = load_profile()
    return {
        "personality_summary": prof.personality_summary[:500],
        "preferences": {
            k: getattr(prof, k)
            for k in [
                "price_sensitivity", "quality_preference", "risk_aversion",
                "research_depth", "brand_trust", "exploration_vs_repeat",
                "review_dependence", "return_preference", "decision_speed",
                "discount_sensitivity",
            ]
        },
        "decision_process": prof.decision_process,
        "semantic_conclusions": [
            {"conclusion": c.get("conclusion", ""), "confidence": c.get("confidence", 0)}
            for c in prof.semantic_conclusions[:6]
        ],
        "shopping_goals": prof.shopping_goals[:400],
    }


# Mount static files — must be LAST
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
