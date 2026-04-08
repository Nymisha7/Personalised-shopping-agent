"""
RL Shopping Agent — Learns to pick products matching the user's personality.

Uses a simple weight-based policy:
  - Maintains feature weights that represent what matters to this user
  - Starts with random/uniform weights (no knowledge)
  - After each episode, updates weights using the personality grader reward
  - Over episodes, converges on the correct feature priorities

The personality_grader acts as the reward function:
  - High personality alignment → positive reward → reinforce those feature weights
  - Low alignment → negative reward → adjust weights away from that pick

Agent Strategy (per episode):
  1. Score all products using current weights × features
  2. Pick the top-scoring product
  3. Get reward from personality grader
  4. Update weights via gradient-like update
  5. If score < threshold, try again with updated weights (within same episode)

Persistence:
  - Weights are saved to memory/agent_weights.json
  - Learning history is saved to memory/learning_log.jsonl
"""

import json
import math
import random
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MEMORY_DIR = Path(__file__).parent / "memory"
WEIGHTS_FILE = MEMORY_DIR / "agent_weights.json"
LEARNING_LOG = MEMORY_DIR / "learning_log.jsonl"

# Feature dimensions the agent learns weights for
FEATURE_KEYS = [
    "price_norm",       # normalized price (0=cheapest, 1=most expensive)
    "rating_norm",      # normalized rating (0-1)
    "reviews_norm",     # normalized review count (0-1)
    "brand_premium",    # 1 if premium brand, 0.5 if mid, 0 if budget/unknown
    "refundable",       # 1 if refundable, 0 if not
    "seller_trusted",   # 1 if official/store, 0 if unknown
    "review_volume",    # log-scaled review count
    "value_ratio",      # rating / price_norm — value for money
]

DEFAULT_WEIGHTS = {k: 0.0 for k in FEATURE_KEYS}  # Start with no knowledge

# Learning hyperparameters
LEARNING_RATE = 0.15
EXPLORATION_RATE_INITIAL = 0.6    # Start with lots of exploration
EXPLORATION_DECAY = 0.85          # Decay per episode
EXPLORATION_MIN = 0.05
SUCCESS_THRESHOLD = 0.75          # Personality score needed to "succeed"
MAX_ATTEMPTS_PER_EPISODE = 8


class RLShoppingAgent:
    """Simple RL agent that learns to pick personality-aligned products."""

    def __init__(self):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.epsilon = EXPLORATION_RATE_INITIAL
        self.episode_count = 0
        self.total_reward = 0.0
        self.success_count = 0
        self._load_weights()

    # ---- Weight persistence ----

    def _load_weights(self):
        """Load saved weights from disk."""
        if WEIGHTS_FILE.exists():
            try:
                data = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
                self.weights = data.get("weights", dict(DEFAULT_WEIGHTS))
                self.epsilon = data.get("epsilon", EXPLORATION_RATE_INITIAL)
                self.episode_count = data.get("episode_count", 0)
                self.total_reward = data.get("total_reward", 0.0)
                self.success_count = data.get("success_count", 0)
            except Exception as e:
                print(f"[rl_agent] Error loading weights: {e}")

    def _save_weights(self):
        """Save weights to disk."""
        try:
            data = {
                "weights": self.weights,
                "epsilon": round(self.epsilon, 4),
                "episode_count": self.episode_count,
                "total_reward": round(self.total_reward, 4),
                "success_count": self.success_count,
                "last_updated": datetime.datetime.now().isoformat(),
            }
            WEIGHTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[rl_agent] Error saving weights: {e}")

    def reset_weights(self):
        """Reset all learned weights (fresh start)."""
        self.weights = dict(DEFAULT_WEIGHTS)
        self.epsilon = EXPLORATION_RATE_INITIAL
        self.episode_count = 0
        self.total_reward = 0.0
        self.success_count = 0
        self._save_weights()

    # ---- Feature extraction ----

    def _extract_features(self, product: Dict, all_products: List[Dict]) -> Dict[str, float]:
        """Extract normalized features from a product."""
        prices = [p.get("price", 0) for p in all_products]
        reviews = [p.get("reviews", 0) for p in all_products]
        min_p, max_p = min(prices), max(prices)
        min_r, max_r = min(reviews), max(reviews)

        price = product.get("price", 0)
        rating = product.get("rating", 3.0)
        review_count = product.get("reviews", 0)
        brand = product.get("brand", "")
        seller = product.get("seller", "")
        refundable = product.get("refundable", True)

        # Normalize price (inverted: lower price = higher score for value shoppers)
        price_norm = 1.0 - ((price - min_p) / max(max_p - min_p, 1.0))
        rating_norm = (rating - 1.0) / 4.0  # 1-5 → 0-1
        reviews_norm = (review_count - min_r) / max(max_r - min_r, 1) if max_r > min_r else 0.5

        # Brand tier
        brand_lower = brand.lower()
        seller_lower = seller.lower()
        brand_premium = 0.5  # default mid
        if any(w in seller_lower for w in ["official", "store"]):
            brand_premium = 0.8
        if any(w in seller_lower for w in ["unknown", "quick", "flash", "random", "budget bazaar"]):
            brand_premium = 0.1

        # Premium brand detection
        if review_count >= 3000 and rating >= 4.5:
            brand_premium = 1.0
        elif review_count < 30 and rating < 3.0:
            brand_premium = 0.0

        seller_trusted = 1.0 if "official" in seller_lower or "store" in seller_lower else 0.0

        review_volume = math.log(max(review_count, 1)) / 10.0  # log scale, ~0-1
        value_ratio = rating_norm / max(1.0 - price_norm + 0.1, 0.1)  # higher = better value

        return {
            "price_norm": round(price_norm, 4),
            "rating_norm": round(rating_norm, 4),
            "reviews_norm": round(reviews_norm, 4),
            "brand_premium": round(brand_premium, 4),
            "refundable": 1.0 if refundable else 0.0,
            "seller_trusted": round(seller_trusted, 4),
            "review_volume": round(min(review_volume, 1.0), 4),
            "value_ratio": round(min(value_ratio, 2.0) / 2.0, 4),  # normalize to 0-1
        }

    def _score_product(self, features: Dict[str, float]) -> float:
        """Score a product using current weights."""
        score = 0.0
        for key in FEATURE_KEYS:
            score += self.weights.get(key, 0.0) * features.get(key, 0.0)
        return score

    # ---- Core RL methods ----

    def pick_product(
        self, products: List[Dict], excluded_ids: set = None
    ) -> Tuple[Dict, Dict[str, float], str]:
        """
        Pick a product using epsilon-greedy policy.

        Returns: (product, features, strategy_used)
        """
        excluded_ids = excluded_ids or set()
        available = [p for p in products if p["id"] not in excluded_ids]
        if not available:
            return None, {}, "no_products"

        # Epsilon-greedy: explore or exploit
        if random.random() < self.epsilon:
            # EXPLORATION: pick randomly
            chosen = random.choice(available)
            features = self._extract_features(chosen, products)
            return chosen, features, "exploration"
        else:
            # EXPLOITATION: pick best according to learned weights
            best_product = None
            best_score = float("-inf")
            best_features = {}

            for p in available:
                feats = self._extract_features(p, products)
                score = self._score_product(feats)
                if score > best_score:
                    best_score = score
                    best_product = p
                    best_features = feats

            return best_product, best_features, "exploitation"

    def update_weights(
        self, features: Dict[str, float], reward: float, personality_score: float
    ):
        """
        Update weights based on reward signal.

        If reward is positive: increase weights for the features present in the product
        If negative: decrease them
        """
        # Simple policy gradient update
        for key in FEATURE_KEYS:
            feat_val = features.get(key, 0.0)
            if feat_val > 0:
                # Update proportional to feature value and reward
                gradient = reward * feat_val
                self.weights[key] += LEARNING_RATE * gradient

        # Decay exploration rate
        self.epsilon = max(EXPLORATION_MIN, self.epsilon * EXPLORATION_DECAY)

        self._save_weights()

    def run_episode(
        self,
        products: List[Dict],
        scored_products: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Run a full autonomous learning episode.

        The agent picks products, the personality_grader scores them,
        and the agent learns from the reward signal.

        Returns list of attempt records.
        """
        self.episode_count += 1
        attempts = []
        excluded_ids = set()
        episode_reward = 0.0

        # Get the ideal product (top personality score) for comparison
        ideal = scored_products[0] if scored_products else None
        ideal_score = ideal["personality_score"] if ideal else 0.5

        for attempt_num in range(1, MAX_ATTEMPTS_PER_EPISODE + 1):
            # Agent picks
            product, features, strategy = self.pick_product(products, excluded_ids)
            if product is None:
                break

            # Get personality score for this product (environment reward)
            personality_score = 0.0
            for sp in scored_products:
                if sp["product"]["id"] == product["id"]:
                    personality_score = sp["personality_score"]
                    break

            # Determine if this is a "success" (close to ideal)
            alignment_ratio = personality_score / max(ideal_score, 0.01)
            is_success = alignment_ratio >= 0.95  # within 5% of ideal

            # Calculate reward
            if is_success:
                # Reward depends on which attempt
                reward_map = {1: 1.0, 2: 0.7, 3: 0.4}
                reward = reward_map.get(attempt_num, 0.2)
            else:
                # Penalty scaled by how far off
                reward = -0.2 - (1.0 - alignment_ratio) * 0.3

            reward = round(reward, 4)
            episode_reward += reward

            # Agent learns from this experience
            self.update_weights(features, reward, personality_score)

            # Build reasoning
            reasoning = self._build_auto_reasoning(
                product, features, strategy, personality_score,
                alignment_ratio, attempt_num, is_success,
            )

            attempt = {
                "attempt": attempt_num,
                "product_id": product["id"],
                "product_name": product["name"],
                "product": product,
                "features": features,
                "strategy": strategy,
                "personality_score": round(personality_score, 4),
                "alignment_ratio": round(alignment_ratio, 4),
                "reward": reward,
                "is_success": is_success,
                "reasoning": reasoning,
                "weights_snapshot": dict(self.weights),
                "epsilon": round(self.epsilon, 4),
            }
            attempts.append(attempt)

            if is_success:
                self.success_count += 1
                break

            # Exclude this product for next attempt
            excluded_ids.add(product["id"])

        self.total_reward += episode_reward
        self._save_weights()
        self._log_episode(attempts, episode_reward)

        return attempts

    def _build_auto_reasoning(
        self, product, features, strategy, p_score,
        alignment, attempt, success,
    ) -> str:
        """Build reasoning text explaining the agent's decision."""
        lines = []

        if strategy == "exploration":
            lines.append(f"🎲 Exploring (ε={self.epsilon:.0%}) — picked randomly to discover new patterns.")
        else:
            lines.append("🧠 Exploiting learned weights — picking what I've learned works best.")

        # Top features that influenced the pick
        weighted = [(k, self.weights[k] * features.get(k, 0)) for k in FEATURE_KEYS]
        weighted.sort(key=lambda x: abs(x[1]), reverse=True)
        top = [w for w in weighted[:3] if abs(w[1]) > 0.01]

        if top:
            influences = ", ".join(
                f"{k.replace('_', ' ')} ({v:+.2f})" for k, v in top
            )
            lines.append(f"Top influences: {influences}")

        lines.append(f"Personality match: {p_score:.0%} (alignment: {alignment:.0%})")

        if success:
            lines.append(f"✅ SUCCESS! Close to ideal product on attempt #{attempt}.")
        else:
            lines.append(f"❌ Not quite right. Updating weights and trying again...")

        return " | ".join(lines)

    def _log_episode(self, attempts: List[Dict], total_reward: float):
        """Log the episode to learning_log.jsonl."""
        try:
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "episode": self.episode_count,
                "attempts_count": len(attempts),
                "total_reward": round(total_reward, 4),
                "success": any(a["is_success"] for a in attempts),
                "final_epsilon": round(self.epsilon, 4),
                "products_tried": [a["product_name"] for a in attempts],
            }
            with open(LEARNING_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[rl_agent] Error logging episode: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent learning statistics."""
        return {
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "success_rate": round(
                self.success_count / max(self.episode_count, 1), 4
            ),
            "total_reward": round(self.total_reward, 4),
            "epsilon": round(self.epsilon, 4),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
        }
