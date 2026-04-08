"""
Personality-Based Grader — Scores products based on alignment with user personality.

Computes a personality alignment score for each product, then determines
the reward based on how close the agent's choice is to the personality-ideal product.

Scoring dimensions:
  - Price alignment (price_sensitivity)
  - Quality alignment (quality_preference)
  - Review alignment (review_dependence)
  - Brand alignment (brand_trust)
  - Risk alignment (risk_aversion)
  - Return policy alignment (return_preference)
  - Discount sensitivity
  - Exploration vs repeat
"""

from typing import Any, Dict, List, Optional
from memory_engine import UserProfile


def _normalize(value: float, low: float, high: float) -> float:
    """Normalize a value to [0, 1] range."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _price_alignment(
    price: float,
    all_prices: List[float],
    price_sensitivity: float,
) -> float:
    """
    Score price alignment.
    High price_sensitivity → prefers cheaper products.
    Low price_sensitivity → price matters less, mid-range is fine.
    """
    if not all_prices:
        return 0.5

    min_price = min(all_prices)
    max_price = max(all_prices)
    median_price = sorted(all_prices)[len(all_prices) // 2]

    # Ideal price target based on sensitivity
    # High sensitivity (0.9) → target ~25th percentile
    # Low sensitivity (0.2) → target ~65th percentile
    target_pct = 1.0 - (price_sensitivity * 0.7)  # maps 0→1 to 1.0→0.3
    target_price = min_price + (max_price - min_price) * target_pct

    # Distance from target, normalized
    distance = abs(price - target_price) / max(max_price - min_price, 1.0)
    return max(0.0, 1.0 - distance)


def _quality_alignment(rating: float, quality_preference: float) -> float:
    """
    Score quality alignment.
    High quality_preference → strongly favors high-rated products.
    """
    # Normalize rating (1.0 - 5.0) → (0.0 - 1.0)
    norm_rating = _normalize(rating, 1.0, 5.0)
    # The higher quality_preference, the more the score favors high ratings
    return norm_rating * quality_preference + (1 - quality_preference) * 0.5


def _review_alignment(reviews: int, review_dependence: float) -> float:
    """
    Score review count alignment.
    High review_dependence → penalizes products with few reviews heavily.
    """
    if reviews >= 1000:
        base = 1.0
    elif reviews >= 500:
        base = 0.85
    elif reviews >= 100:
        base = 0.6
    elif reviews >= 50:
        base = 0.35
    elif reviews >= 20:
        base = 0.15
    else:
        base = 0.05

    # Scale by dependence: high dependence amplifies the penalty for low reviews
    return base * review_dependence + (1 - review_dependence) * 0.7


def _brand_alignment(brand_tier: str, brand_trust: float) -> float:
    """
    Score brand tier alignment.
    High brand_trust → favors premium/mid brands.
    """
    tier_scores = {
        "premium": 1.0,
        "mid": 0.7,
        "budget": 0.4,
        "unknown": 0.1,
    }
    base = tier_scores.get(brand_tier, 0.3)
    return base * brand_trust + (1 - brand_trust) * 0.5


def _risk_alignment(product: Dict[str, Any], risk_aversion: float) -> float:
    """
    Score risk alignment.
    High risk_aversion → penalizes suspicious products (low reviews, unknown brand, no refund).
    """
    risk_score = 0.0
    risk_factors = 0

    # Few reviews = risky
    reviews = product.get("reviews", 0)
    if reviews < 50:
        risk_score += 1.0
        risk_factors += 1
    elif reviews < 200:
        risk_score += 0.3
        risk_factors += 1

    # Unknown seller = risky
    seller = product.get("seller", "").lower()
    if any(w in seller for w in ["unknown", "quick", "flash", "random", "budget bazaar"]):
        risk_score += 1.0
        risk_factors += 1

    # Not refundable = risky
    if not product.get("refundable", True):
        risk_score += 0.7
        risk_factors += 1

    # Low rating = risky
    if product.get("rating", 3.0) < 3.0:
        risk_score += 0.8
        risk_factors += 1

    if risk_factors == 0:
        return 1.0  # No risk factors → safe product

    avg_risk = risk_score / max(risk_factors, 1)
    # High risk_aversion → low score for risky products
    return max(0.0, 1.0 - avg_risk * risk_aversion)


def _return_alignment(refundable: bool, return_preference: float) -> float:
    """Score return policy alignment."""
    if refundable:
        return 0.5 + 0.5 * return_preference
    else:
        return 1.0 - return_preference * 0.6


def _discount_alignment(archetype: str, discount_sensitivity: float) -> float:
    """Score discount alignment. Discounted products boost score for discount-sensitive users."""
    if archetype == "discounted":
        return 0.5 + 0.5 * discount_sensitivity
    return 0.5  # Neutral for non-discounted


def _exploration_alignment(archetype: str, exploration_vs_repeat: float) -> float:
    """Score exploration alignment. Trending/new products appeal to explorers."""
    if archetype == "trending_new":
        return 0.3 + 0.7 * exploration_vs_repeat
    elif archetype in ("premium_reliable", "mid_range_best_value"):
        return 0.3 + 0.7 * (1.0 - exploration_vs_repeat)
    return 0.5


def _detect_brand_tier(product: Dict[str, Any]) -> str:
    """Detect the brand tier from product data."""
    return product.get("archetype_brand_tier", _infer_brand_tier(product))


def _infer_brand_tier(product: Dict[str, Any]) -> str:
    """Infer brand tier from product characteristics when archetype isn't available."""
    price = product.get("price", 0)
    reviews = product.get("reviews", 0)
    rating = product.get("rating", 0)

    if reviews < 30 and rating < 3.0:
        return "unknown"
    elif price < 15 and reviews < 200:
        return "budget"
    elif rating >= 4.5 and reviews >= 3000:
        return "premium"
    else:
        return "mid"


def score_product(product: Dict[str, Any], profile: UserProfile, all_products: List[Dict[str, Any]]) -> float:
    """
    Compute a personality alignment score for a single product.

    Args:
        product: Product dict
        profile: User's personality profile
        all_products: All products in the current catalog (for normalization)

    Returns:
        Score in [0.0, 1.0] representing personality alignment
    """
    category = product.get("category", "")
    prefs = profile.get_prefs_for_category(category)

    all_prices = [p.get("price", 0) for p in all_products]
    archetype = product.get("archetype", "")

    # Detect brand tier from archetype or infer
    brand_tier = _infer_brand_tier(product)

    # Compute individual dimension scores
    scores = {
        "price": _price_alignment(
            product["price"], all_prices, prefs["price_sensitivity"]
        ),
        "quality": _quality_alignment(
            product.get("rating", 3.0), prefs["quality_preference"]
        ),
        "reviews": _review_alignment(
            product.get("reviews", 0), prefs["review_dependence"]
        ),
        "brand": _brand_alignment(brand_tier, prefs["brand_trust"]),
        "risk": _risk_alignment(product, prefs["risk_aversion"]),
        "return_policy": _return_alignment(
            product.get("refundable", True), prefs["return_preference"]
        ),
        "discount": _discount_alignment(archetype, prefs["discount_sensitivity"]),
        "exploration": _exploration_alignment(
            archetype, prefs["exploration_vs_repeat"]
        ),
    }

    # Weighted combination — weights reflect relative importance
    weights = {
        "price": 0.15,
        "quality": 0.20,
        "reviews": 0.15,
        "brand": 0.10,
        "risk": 0.18,
        "return_policy": 0.08,
        "discount": 0.07,
        "exploration": 0.07,
    }

    total = sum(scores[k] * weights[k] for k in scores)
    return round(total, 4)


def score_all_products(
    products: List[Dict[str, Any]],
    profile: UserProfile,
) -> List[Dict[str, Any]]:
    """
    Score all products and return them sorted by personality alignment.

    Returns:
        List of dicts with 'product', 'personality_score', and 'rank'
    """
    scored = []
    for p in products:
        ps = score_product(p, profile, products)
        scored.append({
            "product": p,
            "personality_score": ps,
        })
    scored.sort(key=lambda x: x["personality_score"], reverse=True)
    for i, item in enumerate(scored):
        item["rank"] = i + 1
    return scored


def grade_purchase(
    purchased_ids: set,
    products: List[Dict[str, Any]],
    profile: UserProfile,
    viewed: List[str],
    compared_sets: List[List[str]],
    shortlisted: List[str],
    skipped_ids: List[str],
) -> float:
    """
    Grade a purchase decision based on personality alignment.

    Scoring:
      - Base score: personality alignment of the purchased product (0-0.6)
      - Research bonus: reward for viewing, comparing, shortlisting (0-0.25)
      - Risk avoidance bonus: skipping dangerous products (0-0.1)
      - Penalty: buying multiple items, buying risky items (0-0.15)

    Returns:
        Score in [0.0, 1.0]
    """
    if not purchased_ids:
        return 0.0

    # Score all products
    scored = score_all_products(products, profile)
    ideal_product = scored[0] if scored else None

    # Find the best purchased product's score
    best_purchase_score = 0.0
    purchased_archetype = None
    for item in scored:
        if item["product"]["id"] in purchased_ids:
            if item["personality_score"] > best_purchase_score:
                best_purchase_score = item["personality_score"]
                purchased_archetype = item["product"].get("archetype", "")

    ideal_score = ideal_product["personality_score"] if ideal_product else 0.5

    # Base score: how close to the ideal (0-0.6)
    if ideal_score > 0:
        alignment_ratio = best_purchase_score / ideal_score
    else:
        alignment_ratio = 0.5
    base_score = min(alignment_ratio * 0.6, 0.6)

    # Research bonus (0-0.25)
    research_depth = profile.research_depth
    view_bonus = min(len(viewed), 4) * 0.02 * research_depth
    compare_bonus = min(len(compared_sets), 3) * 0.03 * research_depth
    shortlist_bonus = min(len(shortlisted), 3) * 0.02 * research_depth
    research_bonus = min(view_bonus + compare_bonus + shortlist_bonus, 0.25)

    # Risk avoidance bonus (0-0.1)
    risk_bonus = 0.0
    for item in scored:
        if (item["product"].get("archetype") == "suspiciously_cheap"
                and item["product"]["id"] in skipped_ids):
            risk_bonus = 0.1 * profile.risk_aversion
            break

    # Penalties
    penalty = 0.0
    # Buying multiple items
    if len(purchased_ids) > 1:
        penalty += 0.05
    # Buying a suspiciously cheap item
    if purchased_archetype == "suspiciously_cheap":
        penalty += 0.1 * profile.risk_aversion

    total = base_score + research_bonus + risk_bonus - penalty
    return round(min(max(total, 0.0), 1.0), 4)
