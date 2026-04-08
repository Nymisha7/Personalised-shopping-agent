"""
OpenEnv Inference Runner — Personality-Driven Shopping Agent
=============================================================
Runs the shopping agent through the OpenEnv protocol:
  1. Starts the server (ShoppingEnvironment via create_app)
  2. Connects via ShoppingEnvClient (WebSocket)
  3. Runs 3 tasks (easy → medium → hard) using an LLM or fallback heuristic
  4. Prints results in OpenEnv STDOUT format

STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
  python run_openenv.py
"""

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from openai import OpenAI
from openenv_models import ShoppingAction
from server.shopping_environment import ShoppingEnvironment
from memory_engine import load_profile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://api.openai.com/v1" if os.getenv("OPENAI_API_KEY") else "https://router.huggingface.co/v1",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "shopping_agent"
TEMPERATURE = 0.7
MAX_TOKENS = 300

TASKS = [
    {"name": "quick_pick",  "query": "lip balm",       "difficulty": "easy",   "max_steps": 8,  "product_count": 4},
    {"name": "smart_shop",  "query": "earbuds",         "difficulty": "medium", "max_steps": 12, "product_count": 8},
    {"name": "expert_deal", "query": "laptop backpack", "difficulty": "hard",   "max_steps": 15, "product_count": 12},
]


# ---------------------------------------------------------------------------
# Logging (strict OpenEnv format)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(profile_text: str) -> str:
    return textwrap.dedent(f"""\
You are a shopping agent acting on behalf of a specific user.
Your goal is to buy the product that THIS USER would choose.

{profile_text}

## Your Task
Given the observation (products, personality profile, cart, history),
decide the BEST next action for THIS user. Think about:
- Would this user research more before buying? (check research_depth)
- Would this user pick the cheapest option? (check price_sensitivity)
- Would this user avoid risky products? (check risk_aversion)
- Would this user trust this brand? (check brand_trust)
- Would this user care about reviews? (check review_dependence)

Reply with **only valid JSON**:

{{
  "action_type": "search" | "view_item" | "compare" | "shortlist" |
                 "add_to_cart" | "remove_from_cart" | "buy" | "skip" | "ask_more",
  "item_ids": ["p1", ...],
  "search_query": "query text"
}}

Rules:
- research_depth > 0.7: view 3+ items and compare before buying.
- risk_aversion > 0.7: avoid products with <50 reviews or unknown sellers.
- Do NOT output anything besides the JSON object.
""").strip()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def build_user_prompt(step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    mem_profile = obs_dict.get("memory_profile", {})
    obs_summary = {
        "query": obs_dict.get("query", ""),
        "personality_preferences": {
            k: v for k, v in mem_profile.items()
            if isinstance(v, (int, float)) and 0 <= v <= 1
        },
        "personality_conclusions": mem_profile.get("semantic_conclusions", [])[:4],
        "products": [
            {
                "id": p["id"],
                "name": p["name"],
                "price": p["price"],
                "rating": p["rating"],
                "brand": p["brand"],
                "reviews": p["reviews"],
                "refundable": p["refundable"],
                "seller": p.get("seller", ""),
            }
            for p in obs_dict.get("candidate_products", [])
        ],
        "cart": obs_dict.get("cart", []),
        "shortlisted": obs_dict.get("shortlisted", []),
        "viewed": obs_dict.get("viewed_items", []),
        "feedback": obs_dict.get("feedback", ""),
        "step": obs_dict.get("step_number", step),
        "max_steps": obs_dict.get("max_steps", 15),
    }
    return textwrap.dedent(f"""\
Step {step} | Last reward: {last_reward:.2f}

Observation:
{json.dumps(obs_summary, indent=2)}

Recent history:
{history_block}

Reply with the next action as JSON.""")


def get_agent_action(
    client: Optional[OpenAI],
    system_prompt: str,
    step: int,
    obs_dict: dict,
    last_reward: float,
    history: List[str],
) -> ShoppingAction:
    if client is None:
        return _fallback_action(obs_dict)
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        data = json.loads(text)
        return ShoppingAction(**data)
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        return _fallback_action(obs_dict)


def _fallback_action(obs_dict: dict) -> ShoppingAction:
    """Smart heuristic fallback when the LLM is unavailable."""
    cart = obs_dict.get("cart", [])
    shortlisted = obs_dict.get("shortlisted", [])
    viewed = obs_dict.get("viewed_items", [])
    compared = obs_dict.get("compared_sets", [])
    products = obs_dict.get("candidate_products", [])

    if cart:
        return ShoppingAction(action_type="buy", item_ids=list(cart))
    if shortlisted:
        return ShoppingAction(action_type="add_to_cart", item_ids=[shortlisted[0]])
    if compared:
        flat = []
        for s in compared:
            flat.extend(s)
        unique = list(dict.fromkeys(flat))[:2]
        return ShoppingAction(action_type="shortlist", item_ids=unique)
    if len(viewed) >= 2:
        return ShoppingAction(action_type="compare", item_ids=viewed[:3])
    mid_range = [
        p["id"] for p in products
        if p.get("price", 999) < 150
        and p.get("rating", 0) >= 3.5
        and p.get("reviews", 0) >= 50
    ][:3]
    if mid_range:
        return ShoppingAction(action_type="view_item", item_ids=mid_range)
    if products:
        return ShoppingAction(action_type="view_item", item_ids=[products[0]["id"]])
    return ShoppingAction(action_type="skip")


# ---------------------------------------------------------------------------
# Run one episode DIRECTLY against the OpenEnv Environment (local mode)
# ---------------------------------------------------------------------------
def run_episode_local(llm_client: OpenAI, task: dict) -> tuple:
    """Run a task episode using the OpenEnv Environment directly (no server)."""
    task_name = task["name"]
    query = task["query"]
    max_steps = task["max_steps"]
    product_count = task["product_count"]

    env = ShoppingEnvironment()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Build personality-aware system prompt
    profile = load_profile()
    profile_text = profile.to_prompt_text(category=query)
    system_prompt = build_system_prompt(profile_text)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Use the OpenEnv reset() — returns Observation directly
        obs = env.reset(
            query=query,
            product_count=product_count,
            task_name=task_name,
            max_steps=max_steps,
        )
        obs_dict = obs.model_dump()
        last_reward = 0.0

        for step_num in range(1, max_steps + 1):
            if obs.done:
                break

            action = get_agent_action(
                llm_client, system_prompt, step_num, obs_dict, last_reward, history
            )

            # Use the OpenEnv step() — returns Observation directly
            obs = env.step(action)
            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step_num
            obs_dict = obs.model_dump()
            last_reward = reward

            # Compact action string for logging
            ids_str = ",".join(action.item_ids) if action.item_ids else ""
            if action.search_query:
                action_str = f"{action.action_type}('{action.search_query}')"
            elif ids_str:
                action_str = f"{action.action_type}({ids_str})"
            else:
                action_str = action.action_type

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step_num}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        # Score calculation
        if rewards:
            buy_reward = max((r for r in rewards if r >= 0.3), default=0.0)
            if buy_reward > 0:
                score = buy_reward
            else:
                positive_rewards = sum(r for r in rewards if r > 0)
                score = min(positive_rewards / 2.0, 0.3)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60, flush=True)
    print("OpenEnv Shopping Agent — Direct Local Execution", flush=True)
    print("=" * 60, flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API:   {API_BASE_URL}", flush=True)
    print(f"Key:   {'***' + API_KEY[-4:] if API_KEY else 'NOT SET (using fallback heuristic)'}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

    all_scores = {}
    for task in TASKS:
        _success, _steps, score, _rewards = run_episode_local(llm_client, task)
        all_scores[task["name"]] = score
        print(flush=True)

    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    for t, s in all_scores.items():
        status = "PASS" if s >= 0.5 else "FAIL"
        print(f"  [{status}] {t:16s}: {s:.2f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  {'average':19s}: {avg:.2f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
