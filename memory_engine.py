"""
Memory Engine — Loads and merges the user's personality profile from the memory/ folder.

Reads:
  - memory/personality.md       → personality traits & decision process
  - memory/preferences.json     → numeric preference scores + category overrides
  - memory/semantic_memory.jsonl → learned conclusions about the user
  - memory/episodic_log.jsonl   → past shopping actions
  - memory/projects.md          → current shopping goals

Exposes a single UserProfile dataclass with all merged data.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """Complete user personality profile loaded from memory/ folder."""

    # Core numeric preferences (0.0 – 1.0)
    price_sensitivity: float = 0.5
    quality_preference: float = 0.5
    risk_aversion: float = 0.5
    research_depth: float = 0.5
    brand_trust: float = 0.5
    exploration_vs_repeat: float = 0.5
    review_dependence: float = 0.5
    return_preference: float = 0.5
    decision_speed: float = 0.5
    discount_sensitivity: float = 0.5

    # Category-specific overrides  (e.g. {"electronics": {"risk_aversion": 0.9}})
    category_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Textual personality summary (from personality.md)
    personality_summary: str = ""

    # Decision process steps (parsed from personality.md)
    decision_process: List[str] = field(default_factory=list)

    # Learned conclusions (from semantic_memory.jsonl)
    semantic_conclusions: List[Dict[str, Any]] = field(default_factory=list)

    # Past shopping events (from episodic_log.jsonl)
    episodic_history: List[Dict[str, Any]] = field(default_factory=list)

    # Current shopping goals (from projects.md)
    shopping_goals: str = ""

    def get_prefs_for_category(self, category: str) -> Dict[str, float]:
        """
        Return merged preferences for a specific category.
        Category-specific overrides take precedence over base prefs.
        """
        base = {
            "price_sensitivity": self.price_sensitivity,
            "quality_preference": self.quality_preference,
            "risk_aversion": self.risk_aversion,
            "research_depth": self.research_depth,
            "brand_trust": self.brand_trust,
            "exploration_vs_repeat": self.exploration_vs_repeat,
            "review_dependence": self.review_dependence,
            "return_preference": self.return_preference,
            "decision_speed": self.decision_speed,
            "discount_sensitivity": self.discount_sensitivity,
        }
        # Apply category-specific overrides
        cat_lower = category.lower()
        for cat_key, overrides in self.category_preferences.items():
            if cat_key.lower() in cat_lower or cat_lower in cat_key.lower():
                base.update(overrides)
                break
        return base

    def to_prompt_text(self, category: str = "") -> str:
        """Render the profile as text suitable for an LLM prompt."""
        prefs = self.get_prefs_for_category(category) if category else {}
        lines = [
            "## User Personality Profile",
            "",
            self.personality_summary[:600] if self.personality_summary else "Research-heavy, value-conscious shopper.",
            "",
            "## Numeric Preferences" + (f" (category: {category})" if category else ""),
        ]
        for k, v in (prefs or self.__dict__).items():
            if isinstance(v, float) and 0 <= v <= 1:
                lines.append(f"  - {k}: {v:.2f}")

        if self.semantic_conclusions:
            lines.append("")
            lines.append("## Learned Conclusions")
            for c in self.semantic_conclusions[:6]:
                conf = c.get("confidence", 0)
                lines.append(f"  - {c.get('conclusion', '')} (confidence: {conf:.0%})")

        if self.shopping_goals:
            lines.append("")
            lines.append("## Current Shopping Goals")
            lines.append(self.shopping_goals[:400])

        return "\n".join(lines)


def load_profile(memory_dir: Optional[str] = None) -> UserProfile:
    """
    Load the full user profile from the memory/ directory.
    Falls back to sensible defaults if files are missing.
    """
    if memory_dir is None:
        memory_dir = str(Path(__file__).parent / "memory")
    mem = Path(memory_dir)
    profile = UserProfile()

    # ---- preferences.json ----
    prefs_path = mem / "preferences.json"
    if prefs_path.exists():
        try:
            data = json.loads(prefs_path.read_text(encoding="utf-8"))
            for key in [
                "price_sensitivity", "quality_preference", "risk_aversion",
                "research_depth", "brand_trust", "exploration_vs_repeat",
                "review_dependence", "return_preference", "decision_speed",
                "discount_sensitivity",
            ]:
                if key in data:
                    setattr(profile, key, float(data[key]))
            if "category_preferences" in data:
                profile.category_preferences = data["category_preferences"]
        except Exception as e:
            print(f"[memory_engine] Error loading preferences.json: {e}")

    # ---- personality.md ----
    personality_path = mem / "personality.md"
    if personality_path.exists():
        try:
            text = personality_path.read_text(encoding="utf-8")
            profile.personality_summary = text[:800]
            # Extract decision process steps
            steps = []
            in_process = False
            for line in text.split("\n"):
                stripped = line.strip().lstrip("#").strip()
                if "decision process" in stripped.lower():
                    in_process = True
                    continue
                if in_process and stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
                    steps.append(stripped)
                elif in_process and stripped.startswith("#"):
                    in_process = False
            profile.decision_process = steps
        except Exception as e:
            print(f"[memory_engine] Error loading personality.md: {e}")

    # ---- semantic_memory.jsonl ----
    semantic_path = mem / "semantic_memory.jsonl"
    if semantic_path.exists():
        conclusions = []
        for line in semantic_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    conclusions.append(json.loads(line.strip()))
                except Exception:
                    pass  # skip malformed lines
        profile.semantic_conclusions = conclusions

    # ---- episodic_log.jsonl ----
    episodic_path = mem / "episodic_log.jsonl"
    if episodic_path.exists():
        events = []
        for line in episodic_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    events.append(json.loads(line.strip()))
                except Exception:
                    pass  # skip malformed lines
        profile.episodic_history = events

    # ---- projects.md ----
    projects_path = mem / "projects.md"
    if projects_path.exists():
        try:
            profile.shopping_goals = projects_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[memory_engine] Error loading projects.md: {e}")

    return profile
