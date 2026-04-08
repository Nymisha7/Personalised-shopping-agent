"""
Personalized Shopping Agent Environment
========================================
OpenEnv-compliant RL environment that simulates a real-world shopping session.

The agent must learn the user's shopping style (research-heavy, value-conscious,
risk-averse) and make purchase decisions aligned with the user's personality.

Dynamic product generation: accepts ANY product query (e.g., "lip balm",
"earbuds", "backpack") and generates personality-testing product sets.

Personality-based grading: scores purchases against the user's personality
profile from the memory/ folder.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    Product,
    ShoppingAction,
    ShoppingObservation,
    ShoppingState,
    StepResult,
)
from memory_engine import load_profile, UserProfile
from product_generator import generate_products
from personality_grader import grade_purchase as personality_grade_purchase, score_all_products


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 15
DEFAULT_QUERY = "earbuds"


class ShoppingEnv:
    """
    OpenEnv-compliant shopping environment.

    Dynamic, personality-aware RL environment:
      - Accepts any product query via reset(query)
      - Generates products dynamically via product_generator
      - Loads user profile from memory_engine
      - Grades using personality_grader instead of hardcoded answer keys
      - Logs episodes to episodic_log.jsonl for learning

    Implements reset(), step(action), state(), close().
    """

    def __init__(self, task_name: str = "dynamic"):
        self.task_name = task_name
        self._user_profile: UserProfile = load_profile()

        # Episode state
        self._step_count = 0
        self._max_steps = DEFAULT_MAX_STEPS
        self._done = False
        self._cumulative_reward = 0.0
        self._query = ""
        self._category = ""
        self._cart: List[str] = []
        self._shortlisted: List[str] = []
        self._viewed: List[str] = []
        self._compared_sets: List[List[str]] = []
        self._history: List[str] = []
        self._feedback = ""
        self._skipped_ids: List[str] = []
        self.catalog: List[Dict[str, Any]] = []

        # Personality scoring cache
        self._scored_products: List[Dict[str, Any]] = []

        # User feedback tracking
        self._user_feedback_log: List[Dict[str, Any]] = []

    # ---- OpenEnv lifecycle ------------------------------------------------

    @classmethod
    async def from_docker_image(cls, image_name: str = None):
        return cls()

    async def reset(self, query: str = DEFAULT_QUERY, product_count: int = 8) -> StepResult:
        """Reset the environment with a new product query."""
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._query = query.strip() or DEFAULT_QUERY
        self._category = self._query.lower()
        self._cart = []
        self._shortlisted = []
        self._viewed = []
        self._compared_sets = []
        self._history = []
        self._skipped_ids = []
        self._user_feedback_log = []

        # Reload user profile (may have been updated)
        self._user_profile = load_profile()

        # Generate products dynamically for this query
        self.catalog = generate_products(self._query, count=product_count)

        # Score all products against personality
        self._scored_products = score_all_products(self.catalog, self._user_profile)

        # Find the personality-ideal product for the goal description
        ideal = self._scored_products[0] if self._scored_products else None
        ideal_name = ideal["product"]["name"] if ideal else "unknown"
        ideal_score = ideal["personality_score"] if ideal else 0

        # Research reward scaling based on user's research_depth trait
        research_depth = self._user_profile.research_depth

        self._feedback = (
            f"Welcome! Shopping for: {self._query}\n"
            f"Generated {len(self.catalog)} products.\n"
            f"Goal: Find the product that best matches your personality profile.\n"
            f"The personality-ideal product is '{ideal_name}' "
            f"(alignment score: {ideal_score:.2f}).\n"
            f"Your research depth preference: {research_depth:.0%} — "
            f"{'thorough research expected' if research_depth > 0.7 else 'quick decisions OK'}."
        )

        return StepResult(observation=self._get_obs(), reward=0.0, done=False)

    async def step(self, action: ShoppingAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._get_obs(), reward=0.0, done=True
            )

        self._step_count += 1
        reward = 0.0
        action_log = ""
        research_depth = self._user_profile.research_depth

        atype = action.action_type.lower().strip()

        # ---- search -------------------------------------------------------
        if atype == "search":
            self._query = action.search_query or self._query
            action_log = f"Searched for '{self._query}'"
            self._feedback = f"Found {len(self.catalog)} products for '{self._query}'."
            reward = 0.05 * research_depth

        # ---- view_item ----------------------------------------------------
        elif atype == "view_item":
            if action.item_ids:
                details = []
                for pid in action.item_ids:
                    if pid not in self._viewed:
                        self._viewed.append(pid)
                    prod = self._find_product(pid)
                    if prod:
                        # Include personality score in view details
                        ps = self._get_personality_score(pid)
                        details.append(
                            f"{prod['name']}: ${prod['price']}, "
                            f"{prod['rating']}★, {prod['reviews']} reviews, "
                            f"brand {prod['brand']}, seller {prod['seller']}, "
                            f"refundable={prod['refundable']}, "
                            f"personality_alignment={ps:.2f}"
                        )
                action_log = f"Viewed {len(action.item_ids)} item(s): {action.item_ids}"
                self._feedback = "Details:\n" + "\n".join(
                    f"  - {d}" for d in details
                ) if details else "No matching products found."
                reward = 0.05 * min(len(action.item_ids), 4) * research_depth
            else:
                self._feedback = "view_item requires at least one item_id."

        # ---- compare ------------------------------------------------------
        elif atype == "compare":
            if not action.item_ids or len(action.item_ids) < 2:
                self._feedback = "Compare requires at least 2 item_ids."
            else:
                self._compared_sets.append(list(action.item_ids))
                names = []
                for pid in action.item_ids:
                    p = self._find_product(pid)
                    if p:
                        ps = self._get_personality_score(pid)
                        names.append(
                            f"{p['name']} (${p['price']}, {p['rating']}★, "
                            f"alignment: {ps:.2f})"
                        )
                action_log = f"Compared {len(action.item_ids)} items: {action.item_ids}"
                self._feedback = "Comparison:\n" + "\n".join(
                    f"  - {n}" for n in names
                )
                # More items compared = more reward (encourages research)
                reward = 0.05 * min(len(action.item_ids), 5) * research_depth

        # ---- shortlist ----------------------------------------------------
        elif atype == "shortlist":
            for pid in action.item_ids:
                if pid not in self._shortlisted:
                    self._shortlisted.append(pid)
            action_log = f"Shortlisted {action.item_ids}"
            self._feedback = f"Shortlist now: {self._shortlisted}"
            reward = 0.1 * research_depth

        # ---- add_to_cart --------------------------------------------------
        elif atype == "add_to_cart":
            for pid in action.item_ids:
                if pid not in self._cart:
                    self._cart.append(pid)
            action_log = f"Added to cart: {action.item_ids}"
            self._feedback = f"Cart: {self._cart}"
            reward = 0.1

        # ---- remove_from_cart ---------------------------------------------
        elif atype == "remove_from_cart":
            for pid in action.item_ids:
                if pid in self._cart:
                    self._cart.remove(pid)
            action_log = f"Removed from cart: {action.item_ids}"
            self._feedback = f"Cart: {self._cart}"
            reward = 0.0

        # ---- buy ----------------------------------------------------------
        elif atype == "buy":
            action_log = "Attempted purchase"
            self._done = True
            purchased = set(self._cart + action.item_ids)
            if not purchased:
                self._feedback = "Cannot buy — cart is empty and no item_ids given."
                reward = 0.0
            else:
                reward = self._grade_purchase(purchased)
                # Build detailed feedback
                purchased_names = []
                for pid in purchased:
                    p = self._find_product(pid)
                    if p:
                        ps = self._get_personality_score(pid)
                        purchased_names.append(f"{p['name']} (alignment: {ps:.2f})")
                self._feedback = (
                    f"Purchased: {', '.join(purchased_names)}.\n"
                    f"Final score: {reward:.2f}\n"
                    f"Would you have picked this? Click ✅ Yes or ❌ No below."
                )
                # Log episode
                self._log_episode(purchased, reward)

        # ---- skip ---------------------------------------------------------
        elif atype == "skip":
            if action.item_ids:
                self._skipped_ids.extend(action.item_ids)
                action_log = f"Skipped items: {action.item_ids}"
            else:
                action_log = "Skipped turn"
            self._feedback = "Turn skipped."
            reward = 0.0

        # ---- ask_more -----------------------------------------------------
        elif atype == "ask_more":
            action_log = "Asked for more options"
            self._feedback = "No additional products available in this catalog."
            reward = 0.0

        # ---- unknown action -----------------------------------------------
        else:
            action_log = f"Unknown action: {atype}"
            self._feedback = (
                f"Invalid action_type '{atype}'. Valid: search, view_item, "
                f"compare, shortlist, add_to_cart, remove_from_cart, buy, skip, ask_more."
            )
            reward = -0.1

        if action_log:
            self._history.append(action_log)

        # Penalize running out of steps without buying
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            self._feedback += " Episode ended — max steps reached without purchase."
            reward -= 0.2

        self._cumulative_reward += reward
        return StepResult(
            observation=self._get_obs(),
            reward=round(reward, 4),
            done=self._done,
        )

    async def state(self) -> ShoppingState:
        return ShoppingState(
            task_name=self.task_name,
            difficulty=self.task_name,
            step_count=self._step_count,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            cart=list(self._cart),
            shortlisted=list(self._shortlisted),
            product_query=self._query,
        )

    async def close(self):
        pass

    def record_feedback(self, approved: bool, product_id: str = None, comment: str = None) -> Dict[str, Any]:
        """Record user feedback on the agent's purchase decision."""
        feedback_entry = {
            "approved": approved,
            "product_id": product_id,
            "comment": comment,
            "episode_query": self._query,
            "cumulative_reward": self._cumulative_reward,
        }
        self._user_feedback_log.append(feedback_entry)

        # Adjust reward based on feedback
        feedback_reward = 0.2 if approved else -0.1
        self._cumulative_reward += feedback_reward

        # Log to episodic memory
        self._log_feedback(approved, product_id, comment)

        return {
            "feedback_recorded": True,
            "approved": approved,
            "reward_adjustment": feedback_reward,
            "new_cumulative_reward": round(self._cumulative_reward, 4),
        }

    def get_profile_summary(self) -> Dict[str, Any]:
        """Return the user's personality profile summary."""
        profile = self._user_profile
        return {
            "personality_summary": profile.personality_summary[:400],
            "preferences": profile.get_prefs_for_category(self._category),
            "decision_process": profile.decision_process,
            "semantic_conclusions": [
                {
                    "conclusion": c.get("conclusion", ""),
                    "confidence": c.get("confidence", 0),
                }
                for c in profile.semantic_conclusions[:6]
            ],
            "shopping_goals": profile.shopping_goals[:300],
            "episode_history_count": len(profile.episodic_history),
        }

    # ---- helpers ----------------------------------------------------------

    def _find_product(self, pid: str) -> Optional[Dict[str, Any]]:
        for p in self.catalog:
            if p["id"] == pid:
                return p
        return None

    def _get_personality_score(self, pid: str) -> float:
        """Get the personality alignment score for a product."""
        for item in self._scored_products:
            if item["product"]["id"] == pid:
                return item["personality_score"]
        return 0.0

    def _get_obs(self) -> ShoppingObservation:
        history_text = "\n".join(self._history[-6:]) if self._history else "No actions yet."
        profile = self._user_profile
        prefs = profile.get_prefs_for_category(self._category)

        return ShoppingObservation(
            query=self._query,
            category=self._category,
            candidate_products=self.catalog,
            memory_profile={
                "goal": (
                    f"Find the best {self._query} that matches your personality: "
                    f"research-heavy ({prefs.get('research_depth', 0.5):.0%}), "
                    f"value-conscious ({prefs.get('price_sensitivity', 0.5):.0%}), "
                    f"quality-focused ({prefs.get('quality_preference', 0.5):.0%})."
                ),
                **prefs,
                "semantic_conclusions": [
                    c.get("conclusion", "")
                    for c in profile.semantic_conclusions[:6]
                ],
                "personality_summary": profile.personality_summary[:300],
            },
            cart=list(self._cart),
            shortlisted=list(self._shortlisted),
            viewed_items=list(self._viewed),
            compared_sets=[list(s) for s in self._compared_sets],
            history_summary=history_text,
            feedback=self._feedback,
            step_number=self._step_count,
            max_steps=self._max_steps,
        )

    # ---- grader -----------------------------------------------------------

    def _grade_purchase(self, purchased_ids: set) -> float:
        """
        Personality-based grader that returns a score in [0.0, 1.0].
        Uses personality_grader to score based on user profile alignment.
        """
        return personality_grade_purchase(
            purchased_ids=purchased_ids,
            products=self.catalog,
            profile=self._user_profile,
            viewed=self._viewed,
            compared_sets=self._compared_sets,
            shortlisted=self._shortlisted,
            skipped_ids=self._skipped_ids,
        )

    # ---- episodic logging -------------------------------------------------

    def _log_episode(self, purchased_ids: set, reward: float):
        """Log the purchase decision to episodic_log.jsonl."""
        try:
            import datetime
            log_path = Path(__file__).parent / "memory" / "episodic_log.jsonl"
            purchased_products = []
            for pid in purchased_ids:
                p = self._find_product(pid)
                if p:
                    purchased_products.append({
                        "id": p["id"],
                        "name": p["name"],
                        "price": p["price"],
                        "brand": p["brand"],
                    })
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event": "agent_purchase",
                "query": self._query,
                "purchased": purchased_products,
                "reward": round(reward, 4),
                "steps": self._step_count,
                "viewed_count": len(self._viewed),
                "compared_count": len(self._compared_sets),
                "shortlisted_count": len(self._shortlisted),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[shopping_env] Error logging episode: {e}")

    def _log_feedback(self, approved: bool, product_id: str = None, comment: str = None):
        """Log user feedback to episodic_log.jsonl."""
        try:
            import datetime
            log_path = Path(__file__).parent / "memory" / "episodic_log.jsonl"
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event": "user_feedback",
                "query": self._query,
                "product_id": product_id,
                "approved": approved,
                "comment": comment,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[shopping_env] Error logging feedback: {e}")
