"""
FastAPI app for the Shopping Agent — powered by OpenEnv.

Uses openenv.core.env_server.http_server.create_app() to expose the
ShoppingEnvironment over HTTP + WebSocket endpoints that any EnvClient
(including ShoppingEnvClient) can consume.

Endpoints auto-provided by OpenEnv:
  POST /reset       → Reset environment
  POST /step        → Execute action
  GET  /state       → Current episode state
  GET  /health      → Health check
  GET  /schema      → Action/Observation schemas
  WS   /ws          → WebSocket persistent session

Custom endpoints added below:
  GET  /             → Web UI
  GET  /profile      → User personality profile
"""

import os
import sys
from pathlib import Path

import uvicorn
from pydantic import BaseModel, Field

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server.http_server import create_app
from openenv_models import ShoppingAction, ShoppingObservation
from server.shopping_environment import ShoppingEnvironment
from memory_engine import load_profile
from rl_agent import RLShoppingAgent

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_agent = RLShoppingAgent()
_ui_env: ShoppingEnvironment | None = None


class UIRestBody(BaseModel):
    """Payload for the custom UI reset endpoint."""

    task: str | None = Field(default=None)
    query: str | None = Field(default=None)

# --- Create the OpenEnv app -------------------------------------------------
# Pass the CLASS (factory), not an instance — create_app creates per-session.
app = create_app(
    ShoppingEnvironment,
    ShoppingAction,
    ShoppingObservation,
    env_name="shopping_agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Custom endpoints -------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse(content={"message": "Shopping Agent OpenEnv server is running."})


@app.get("/profile")
async def profile():
    """Return the loaded user personality profile summary."""
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


@app.post("/ui/reset")
async def ui_reset(body: UIRestBody) -> dict:
    """Reset a dedicated UI session with the richer legacy payload."""
    global _ui_env

    _ui_env = ShoppingEnvironment()
    observation = _ui_env.reset(task=body.task, query=body.query or "")
    profile_prefs = _ui_env._user_profile.get_prefs_for_category(observation.category)

    return {
        "query": observation.query,
        "product_count": len(_ui_env.catalog),
        "products": list(_ui_env.catalog),
        "scored_products": [
            {
                "id": item["product"]["id"],
                "name": item["product"]["name"],
                "score": item["personality_score"],
                "rank": item["rank"],
            }
            for item in _ui_env._scored_products
        ],
        "personality_traits": profile_prefs,
        "ideal_product": (
            _ui_env._scored_products[0]["product"]["name"]
            if _ui_env._scored_products
            else None
        ),
        "ideal_score": (
            _ui_env._scored_products[0]["personality_score"]
            if _ui_env._scored_products
            else 0.0
        ),
    }


@app.post("/auto-run")
async def auto_run() -> dict:
    """Run the autonomous RL helper flow used by the bundled web UI."""
    if _ui_env is None or not _ui_env.catalog:
        return JSONResponse(
            status_code=400,
            content={"error": "No active UI session. Start with /ui/reset first."},
        )

    attempts = _agent.run_episode(
        products=_ui_env.catalog,
        scored_products=_ui_env._scored_products,
    )
    stats = _agent.get_stats()
    ideal = _ui_env._scored_products[0] if _ui_env._scored_products else None

    return {
        "attempts": attempts,
        "episode_number": stats["episode_count"],
        "total_episode_reward": round(sum(a["reward"] for a in attempts), 4),
        "success": any(a["is_success"] for a in attempts),
        "success_attempt": next(
            (a["attempt"] for a in attempts if a["is_success"]),
            None,
        ),
        "ideal_product": (
            {
                "name": ideal["product"]["name"],
                "score": ideal["personality_score"],
                "id": ideal["product"]["id"],
            }
            if ideal
            else None
        ),
        "agent_stats": stats,
    }


@app.get("/agent-stats")
async def agent_stats() -> dict:
    """Return persisted RL helper stats for the web UI."""
    return _agent.get_stats()


@app.post("/agent-reset")
async def agent_reset() -> dict:
    """Reset persisted RL helper weights for the web UI."""
    _agent.reset_weights()
    return {
        "message": "Agent weights reset. Learning starts from scratch.",
        "stats": _agent.get_stats(),
    }


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    """Run the server directly."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
