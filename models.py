"""
Type-safe Pydantic models for the Personalized Shopping Agent Environment.

Defines the Action, Observation, and State contracts used across the
server, client, and inference script.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ---------------------------------------------------------------------------
# Product schema
# ---------------------------------------------------------------------------
class Product(BaseModel):
    """A single product in the catalog."""
    id: str
    name: str
    price: float
    rating: float
    brand: str
    reviews: int
    category: str
    seller: str
    refundable: bool
    features: List[str] = []
    personality_score: Optional[float] = Field(
        default=None,
        description="Personality alignment score (0.0-1.0), set by the grader",
    )


# ---------------------------------------------------------------------------
# Action: what the agent sends
# ---------------------------------------------------------------------------
class ShoppingAction(BaseModel):
    """
    An action the agent can take in the shopping environment.

    Supported action_types:
      search        – search catalog (requires search_query)
      view_item     – view details of one item (requires item_ids[0])
      compare       – compare 2+ items side-by-side (requires item_ids)
      shortlist     – mark items for later (requires item_ids)
      add_to_cart   – add items to cart (requires item_ids)
      remove_from_cart – remove items from cart (requires item_ids)
      buy           – purchase items in cart + item_ids
      skip          – do nothing this turn
      ask_more      – request more product options
    """
    action_type: str = Field(
        ...,
        description="One of: search, view_item, compare, shortlist, "
                    "add_to_cart, remove_from_cart, buy, skip, ask_more",
    )
    item_ids: List[str] = Field(
        default_factory=list,
        description="Product IDs involved in the action",
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Query string when action_type is 'search'",
    )


# ---------------------------------------------------------------------------
# Reset request: accepts a product query
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    """Body for the /reset endpoint — accepts a free-form product query."""
    query: str = Field(
        default="earbuds",
        description="What product to shop for (e.g., 'lip balm', 'earbuds', 'backpack')",
    )


# ---------------------------------------------------------------------------
# User feedback: reward signal from the user
# ---------------------------------------------------------------------------
class UserFeedback(BaseModel):
    """User confirms or rejects the agent's purchase decision."""
    approved: bool = Field(
        ...,
        description="True if the user agrees with the agent's choice, False otherwise",
    )
    product_id: Optional[str] = Field(
        default=None,
        description="The product ID the agent chose (for logging)",
    )
    comment: Optional[str] = Field(
        default=None,
        description="Optional user comment on the decision",
    )


# ---------------------------------------------------------------------------
# Observation: what the environment returns
# ---------------------------------------------------------------------------
class ShoppingObservation(BaseModel):
    """What the agent observes after each step."""
    query: str = Field("", description="Current search query")
    category: str = Field("", description="Current product category")
    candidate_products: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Products currently visible to the agent",
    )
    memory_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="User's preferences, personality traits, and semantic memory",
    )
    cart: List[str] = Field(
        default_factory=list,
        description="Product IDs in the cart",
    )
    shortlisted: List[str] = Field(
        default_factory=list,
        description="Product IDs on the shortlist",
    )
    viewed_items: List[str] = Field(
        default_factory=list,
        description="Product IDs the agent has viewed in detail",
    )
    compared_sets: List[List[str]] = Field(
        default_factory=list,
        description="Sets of item IDs that were compared",
    )
    history_summary: str = Field(
        "", description="Human-readable summary of recent actions",
    )
    feedback: str = Field(
        "", description="Environment feedback on the last action",
    )
    step_number: int = Field(0, description="Current step in the episode")
    max_steps: int = Field(15, description="Max steps before episode ends")


# ---------------------------------------------------------------------------
# State: episode metadata
# ---------------------------------------------------------------------------
class ShoppingState(BaseModel):
    """Episode-level metadata returned by state()."""
    task_name: str = ""
    difficulty: str = ""
    step_count: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    cart: List[str] = Field(default_factory=list)
    shortlisted: List[str] = Field(default_factory=list)
    product_query: str = Field(
        default="",
        description="The current product query for this episode",
    )


# ---------------------------------------------------------------------------
# StepResult: wrapper returned by reset() and step()
# ---------------------------------------------------------------------------
class StepResult(BaseModel):
    """Result of a reset() or step() call."""
    observation: ShoppingObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
