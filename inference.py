"""
Inference Script — Personality-Driven Shopping Agent Environment
================================================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Runs 3 tasks (easy → medium → hard) and produces baseline scores.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from models import ShoppingAction
from shopping_env import ShoppingEnv
from memory_engine import load_profile

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (reads from env vars, with defaults)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "shopping_agent"
TEMPERATURE = 0.7
MAX_TOKENS = 300

# 3 tasks: easy → medium → hard
TASKS = [
    {"name": "quick_pick",  "query": "lip balm",        "difficulty": "easy",   "max_steps": 8,  "product_count": 4},
    {"name": "smart_shop",  "query": "earbuds",          "difficulty": "medium", "max_steps": 12, "product_count": 8},
    {"name": "expert_deal", "query": "laptop backpack",  "difficulty": "hard",   "max_steps": 15, "product_count": 12},
]


# ---------------------------------------------------------------------------
# Logging helpers (strict format — do not change field order)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt: built dynamically from the user's personality profile
# ---------------------------------------------------------------------------
def build_system_prompt(profile_text: str) -> str:
    """Build the system prompt with the full user personality profile."""
    return textwrap.dedent(f"""\
You are a shopping agent acting on behalf of a specific user.
Your goal is to buy the product that THIS USER would choose — not the
generically "best" product.

{profile_text}

## Your Task
Given the current observation (products, personality profile, cart, history),
decide the BEST next action for THIS specific user. Think about:
- Would this user research more before buying? (check research_depth)
- Would this user pick the cheapest option? (check price_sensitivity)
- Would this user avoid risky products? (check risk_aversion)
- Would this user trust this brand? (check brand_trust)
- Would this user care about reviews? (check review_dependence)

You MUST reply with **only valid JSON** matching this schema:

{{
  "action_type": "search" | "view_item" | "compare" | "shortlist" |
                 "add_to_cart" | "remove_from_cart" | "buy" | "skip" | "ask_more",
  "item_ids": ["p1", ...],       // optional, list of product IDs
  "search_query": "query text"   // optional, only for search
}}

Rules:
- Always think about what THIS USER would do based on their personality traits.
- Research depth matters: if research_depth > 0.7, view 3+ items and compare before buying.
- Risk aversion matters: if risk_aversion > 0.7, avoid products with <50 reviews or unknown sellers.
- The user's personality determines which product is "right", not just price or ratings.
- Do NOT output anything besides the JSON object.
""").strip()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def build_user_prompt(
    step: int,
    obs_dict: dict,
    last_reward: float,
    history: List[str],
) -> str:
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

Think about what this specific user would do based on their personality traits.
Reply with the next action as JSON.""")


def get_agent_action(
    client: OpenAI,
    system_prompt: str,
    step: int,
    obs_dict: dict,
    last_reward: float,
    history: List[str],
) -> ShoppingAction:
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
        # Strip markdown fences if present
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
    """
    Smart heuristic fallback when the LLM is unavailable.
    Progresses through the shopping workflow based on current state.
    """
    cart = obs_dict.get("cart", [])
    shortlisted = obs_dict.get("shortlisted", [])
    viewed = obs_dict.get("viewed_items", [])
    compared = obs_dict.get("compared_sets", [])
    products = obs_dict.get("candidate_products", [])

    # If cart has items, buy
    if cart:
        return ShoppingAction(action_type="buy", item_ids=list(cart))

    # If items are shortlisted, add best one to cart
    if shortlisted:
        return ShoppingAction(action_type="add_to_cart", item_ids=[shortlisted[0]])

    # If items have been compared, shortlist top 2
    if compared:
        flat = []
        for s in compared:
            flat.extend(s)
        unique = list(dict.fromkeys(flat))[:2]
        return ShoppingAction(action_type="shortlist", item_ids=unique)

    # If items have been viewed, compare them
    if len(viewed) >= 2:
        return ShoppingAction(action_type="compare", item_ids=viewed[:3])

    # Otherwise view some mid-range products (avoid suspiciously cheap)
    mid_range = [
        p["id"] for p in products
        if p.get("price", 999) < 150
        and p.get("rating", 0) >= 3.5
        and p.get("reviews", 0) >= 50
    ][:3]
    if mid_range:
        return ShoppingAction(action_type="view_item", item_ids=mid_range)

    # Last resort
    if products:
        return ShoppingAction(action_type="view_item", item_ids=[products[0]["id"]])

    return ShoppingAction(action_type="skip")


# ---------------------------------------------------------------------------
# Run one episode for a task
# ---------------------------------------------------------------------------
async def run_episode(
    client: OpenAI, task: dict
) -> tuple:
    """Run a single task episode. Returns (success, steps, score, rewards)."""
    task_name = task["name"]
    query = task["query"]
    max_steps = task["max_steps"]
    product_count = task["product_count"]

    env = ShoppingEnv(task_name=task_name)
    env._max_steps = max_steps
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
        result = await env.reset(query=query, product_count=product_count)
        obs_dict = result.observation.model_dump()
        last_reward = 0.0

        for step_num in range(1, max_steps + 1):
            if result.done:
                break

            action = get_agent_action(
                client, system_prompt, step_num, obs_dict, last_reward, history
            )
            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step_num
            obs_dict = result.observation.model_dump()
            last_reward = reward

            # Build compact action string for logging
            ids_str = ",".join(action.item_ids) if action.item_ids else ""
            if action.search_query:
                action_str = f"{action.action_type}('{action.search_query}')"
            elif ids_str:
                action_str = f"{action.action_type}({ids_str})"
            else:
                action_str = action.action_type

            log_step(
                step=step_num,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )
            history.append(
                f"Step {step_num}: {action_str} -> reward {reward:+.2f}"
            )

            if done:
                break

        # Score calculation:
        # The buy grader reward (personality alignment) IS the score
        # If no buy, partial credit from research steps
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
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return success, steps_taken, score, rewards


# ---------------------------------------------------------------------------
# Main: run all 3 tasks
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    for task in TASKS:
        _success, _steps, score, _rewards = await run_episode(client, task)
        all_scores[task["name"]] = score
        print(flush=True)  # blank line between tasks

    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    for t, s in all_scores.items():
        status = "✅" if s >= 0.5 else "❌"
        print(f"  {status} {t:16s}: {s:.2f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  {'average':19s}: {avg:.2f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
