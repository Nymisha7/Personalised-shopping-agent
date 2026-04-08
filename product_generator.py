"""
Product Generator — Dynamically generates realistic product catalogs for ANY category.

Each product set is designed to test different personality dimensions:
  - Suspiciously cheap   → tests risk_aversion, review_dependence
  - Budget value         → tests price_sensitivity, quality_preference
  - Mid-range best-value → tests value-over-cheapest tendency
  - Premium reliable     → tests brand_trust, quality_preference
  - Luxury overpriced    → tests price_sensitivity vs brand_trust
  - Popular but mixed    → tests review_dependence, risk_aversion
  - Trending new         → tests exploration_vs_repeat
  - Discounted           → tests discount_sensitivity
"""

import random
import hashlib
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Category attribute templates
# ---------------------------------------------------------------------------
CATEGORY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "lip balm": {
        "features_pool": [
            "SPF 15", "SPF 30", "moisturizing", "tinted", "organic",
            "vitamin E", "beeswax", "petroleum-free", "aloe vera",
            "long-lasting", "matte finish", "glossy finish", "medicated",
            "cocoa butter", "shea butter", "mint flavor", "vanilla flavor",
            "strawberry flavor", "unscented", "travel-size",
        ],
        "brands": {
            "premium": ["Burt's Bees", "Laneige", "Fresh", "Tatcha"],
            "mid": ["Nivea", "Vaseline", "Maybelline", "Neutrogena"],
            "budget": ["Himalaya", "Lakme", "Colorbar", "Generic Beauty"],
            "unknown": ["BeautyDeals", "LipGlow99", "NoName Cosmetics"],
        },
        "price_range": (1.50, 35.00),
        "unit": "$",
    },
    "earbuds": {
        "features_pool": [
            "ANC", "Bluetooth 5.3", "Bluetooth 5.0", "Bluetooth 4.0",
            "wireless charging", "IPX5", "IP67", "Hi-Res Audio",
            "spatial audio", "touch controls", "in-ear", "open-ear",
            "ear hooks", "multipoint", "USB-C", "low latency",
            "30h battery", "24h battery", "16h battery", "10h battery",
            "8h battery", "6h battery", "4h battery", "2h battery",
        ],
        "brands": {
            "premium": ["Sony", "Bose", "Apple", "Sennheiser"],
            "mid": ["JBL", "SoundWave", "Anker", "OnePlus"],
            "budget": ["Boat", "Realme", "pTron", "FitSound"],
            "unknown": ["GenericTech", "NoName Audio", "QuickBuds"],
        },
        "price_range": (4.99, 299.99),
        "unit": "$",
    },
    "backpack": {
        "features_pool": [
            "laptop compartment", "waterproof", "water-resistant",
            "USB charging port", "anti-theft", "padded straps",
            "multiple compartments", "breathable back panel",
            "reflective strips", "expandable", "lightweight",
            "trolley strap", "bottle holder", "key clip",
            "RFID blocking pocket", "rain cover included",
        ],
        "brands": {
            "premium": ["Samsonite", "Osprey", "North Face", "Tumi"],
            "mid": ["American Tourister", "Wildcraft", "Skybags", "SwissGear"],
            "budget": ["F Gear", "Safari", "Wesley", "Gear"],
            "unknown": ["BagDeal123", "NoName Bags", "CheapPack"],
        },
        "price_range": (8.99, 189.99),
        "unit": "$",
    },
    "mixer grinder": {
        "features_pool": [
            "750W motor", "500W motor", "1000W motor",
            "3 jars", "4 jars", "stainless steel blades",
            "overload protection", "anti-skid feet", "pulse function",
            "5-year warranty", "2-year warranty", "1-year warranty",
            "low noise", "food processor attachment", "juicer attachment",
            "BPA-free jars", "copper motor winding",
        ],
        "brands": {
            "premium": ["Preethi", "Philips", "Bosch", "KitchenAid"],
            "mid": ["Bajaj", "Butterfly", "Prestige", "Morphy Richards"],
            "budget": ["Pigeon", "Inalsa", "Lifelong", "Orient"],
            "unknown": ["KitchenDeals", "NoName Appliance", "CheapMix"],
        },
        "price_range": (15.99, 149.99),
        "unit": "$",
    },
    "headphones": {
        "features_pool": [
            "ANC", "over-ear", "on-ear", "Bluetooth 5.3",
            "wired option", "40mm drivers", "50mm drivers",
            "Hi-Res Audio", "spatial audio", "foldable",
            "memory foam cushions", "30h battery", "60h battery",
            "fast charging", "multipoint", "detachable cable",
            "built-in mic", "noise isolation", "lightweight",
        ],
        "brands": {
            "premium": ["Sony", "Bose", "Sennheiser", "Bang & Olufsen"],
            "mid": ["JBL", "Audio-Technica", "AKG", "Jabra"],
            "budget": ["Boat", "OneOdio", "Soundcore", "Mpow"],
            "unknown": ["GenericAudio", "BassMax99", "CheapSound"],
        },
        "price_range": (9.99, 399.99),
        "unit": "$",
    },
    "water bottle": {
        "features_pool": [
            "insulated", "double-wall vacuum", "BPA-free",
            "leak-proof lid", "straw lid", "wide mouth",
            "24h cold / 12h hot", "stainless steel", "tritan plastic",
            "500ml", "750ml", "1L", "carrying loop",
            "dishwasher safe", "powder coated", "infuser included",
        ],
        "brands": {
            "premium": ["Hydro Flask", "Yeti", "S'well", "Klean Kanteen"],
            "mid": ["CamelBak", "Nalgene", "Contigo", "Thermos"],
            "budget": ["Milton", "Cello", "Tupperware", "Signoraware"],
            "unknown": ["H2ODeals", "NoName Flask", "CheapBottle"],
        },
        "price_range": (3.99, 49.99),
        "unit": "$",
    },
}

# Default template used for any unrecognized category
DEFAULT_TEMPLATE: Dict[str, Any] = {
    "features_pool": [
        "high quality", "durable", "lightweight", "compact",
        "eco-friendly", "premium material", "ergonomic design",
        "value pack", "limited edition", "customer favorite",
        "easy to use", "portable", "long-lasting", "versatile",
    ],
    "brands": {
        "premium": ["PremiumBrand", "LuxCo", "EliteMaker", "TopTier"],
        "mid": ["MidRange Inc", "ValueBrand", "TrustMaker", "ReliableCo"],
        "budget": ["BudgetChoice", "AffordableCo", "EasyBuy", "SmartSave"],
        "unknown": ["Unknown Seller", "NoName Store", "CheapDeals"],
    },
    "price_range": (5.00, 200.00),
    "unit": "$",
}


# ---------------------------------------------------------------------------
# Product archetypes
# ---------------------------------------------------------------------------
ARCHETYPES = [
    {
        "key": "suspiciously_cheap",
        "name_prefix": "Ultra-Cheap",
        "price_pct": 0.05,   # 5% of range
        "rating": (1.8, 2.5),
        "reviews": (3, 25),
        "brand_tier": "unknown",
        "refundable": False,
        "seller_type": "unknown",
        "feature_count": (1, 2),
    },
    {
        "key": "budget_value",
        "name_prefix": "Budget",
        "price_pct": 0.18,
        "rating": (3.5, 4.0),
        "reviews": (200, 600),
        "brand_tier": "budget",
        "refundable": True,
        "seller_type": "store",
        "feature_count": (2, 4),
    },
    {
        "key": "mid_range_best_value",
        "name_prefix": "Mid-Range",
        "price_pct": 0.35,
        "rating": (4.2, 4.5),
        "reviews": (800, 2500),
        "brand_tier": "mid",
        "refundable": True,
        "seller_type": "official",
        "feature_count": (3, 5),
    },
    {
        "key": "premium_reliable",
        "name_prefix": "Premium",
        "price_pct": 0.65,
        "rating": (4.5, 4.8),
        "reviews": (3000, 8000),
        "brand_tier": "premium",
        "refundable": True,
        "seller_type": "official",
        "feature_count": (4, 6),
    },
    {
        "key": "luxury_overpriced",
        "name_prefix": "Luxury",
        "price_pct": 0.92,
        "rating": (4.7, 4.9),
        "reviews": (5000, 15000),
        "brand_tier": "premium",
        "refundable": True,
        "seller_type": "official",
        "feature_count": (5, 7),
    },
    {
        "key": "popular_but_mixed",
        "name_prefix": "Popular",
        "price_pct": 0.40,
        "rating": (3.2, 3.8),
        "reviews": (4000, 12000),
        "brand_tier": "mid",
        "refundable": True,
        "seller_type": "store",
        "feature_count": (3, 5),
    },
    {
        "key": "trending_new",
        "name_prefix": "Trending",
        "price_pct": 0.50,
        "rating": (4.3, 4.6),
        "reviews": (15, 80),
        "brand_tier": "mid",
        "refundable": True,
        "seller_type": "store",
        "feature_count": (4, 6),
    },
    {
        "key": "discounted",
        "name_prefix": "Sale",
        "price_pct": 0.30,    # discounted from ~60% to ~30%
        "rating": (4.0, 4.4),
        "reviews": (500, 2000),
        "brand_tier": "mid",
        "refundable": True,
        "seller_type": "official",
        "feature_count": (3, 5),
    },
]


def _get_template(category: str) -> Dict[str, Any]:
    """Get the category template, falling back to default."""
    cat_lower = category.lower().strip()
    for key, template in CATEGORY_TEMPLATES.items():
        if key in cat_lower or cat_lower in key:
            return template
    return DEFAULT_TEMPLATE


def _make_seller(brand: str, seller_type: str) -> str:
    """Generate a seller name based on the brand and seller type."""
    if seller_type == "official":
        return f"{brand} Official"
    elif seller_type == "store":
        return f"{brand} Store"
    else:
        unknown_sellers = [
            "Unknown Marketplace", "Quick Deals", "Flash Sales",
            "Random Seller", "Budget Bazaar",
        ]
        return random.choice(unknown_sellers)


def _deterministic_seed(query: str, salt: str = "") -> int:
    """Generate a deterministic seed from the query for reproducible results."""
    h = hashlib.md5((query + salt).encode()).hexdigest()
    return int(h[:8], 16)


def generate_products(
    query: str,
    count: int = 8,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a realistic set of products for any category query.

    Args:
        query: Product search query (e.g., "lip balm", "wireless earbuds")
        count: Number of products to generate (default 8)
        seed: Optional random seed for reproducibility

    Returns:
        List of product dicts matching the Product model schema
    """
    if seed is None:
        seed = _deterministic_seed(query)
    rng = random.Random(seed)

    template = _get_template(query)
    features_pool = list(template["features_pool"])
    brands = template["brands"]
    price_min, price_max = template["price_range"]
    price_span = price_max - price_min

    # Use up to `count` archetypes
    archetypes = ARCHETYPES[:count]
    products: List[Dict[str, Any]] = []

    # Clean up the query for product naming
    category_name = query.strip().title()

    for i, arch in enumerate(archetypes):
        # Select brand
        brand_list = brands.get(arch["brand_tier"], brands["mid"])
        brand = rng.choice(brand_list)

        # Calculate price
        base_price = price_min + (price_span * arch["price_pct"])
        price_jitter = rng.uniform(-0.05, 0.05) * price_span
        price = round(max(price_min, base_price + price_jitter), 2)

        # Rating
        rating = round(rng.uniform(*arch["rating"]), 1)

        # Reviews
        reviews = rng.randint(*arch["reviews"])

        # Features
        n_features = rng.randint(*arch["feature_count"])
        rng.shuffle(features_pool)
        features = features_pool[:n_features]

        # Seller
        seller = _make_seller(brand, arch["seller_type"])

        # Build product name
        name = f"{arch['name_prefix']} {category_name}"
        if arch["key"] == "discounted":
            name = f"{category_name} (Was ${round(price * 1.8, 2)}, Now Sale!)"

        product = {
            "id": f"p{i + 1}",
            "name": name,
            "price": price,
            "rating": rating,
            "brand": brand,
            "reviews": reviews,
            "category": query.lower().strip(),
            "seller": seller,
            "refundable": arch["refundable"],
            "features": features,
            "archetype": arch["key"],  # kept for grading reference
        }
        products.append(product)

    return products


def get_supported_categories() -> List[str]:
    """Return list of categories with specialized templates."""
    return list(CATEGORY_TEMPLATES.keys())
