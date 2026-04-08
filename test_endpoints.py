"""Quick test of all OpenEnv endpoints."""
import requests
import json

BASE = "http://localhost:7860"

# Test 1: Health
print("=== /health ===")
r = requests.get(f"{BASE}/health")
print(f"  Status: {r.status_code} -> {r.json()}")

# Test 2: Reset with easy task
print("\n=== /reset (quick_pick) ===")
r = requests.post(f"{BASE}/reset", json={"task": "quick_pick"})
data = r.json()
info = data["info"]
print(f"  Task: {info['task']}, Difficulty: {info['difficulty']}")
print(f"  Products: {info['product_count']}, Max steps: {info['max_steps']}")
print(f"  Ideal product: {info['ideal_product']} (score: {info['ideal_score']:.2f})")
products = data["observation"]["candidate_products"]
pids = [p["id"] for p in products]
print(f"  Product IDs: {pids}")

# Test 3: Step - view items
print("\n=== /step (view_item) ===")
r = requests.post(f"{BASE}/step", json={"action_type": "view_item", "item_ids": pids[:2]})
d = r.json()
print(f"  Reward: {d['reward']}, Done: {d['done']}, Step: {d['info']['step_count']}")

# Test 4: Step - compare
print("\n=== /step (compare) ===")
r = requests.post(f"{BASE}/step", json={"action_type": "compare", "item_ids": pids[:3]})
d = r.json()
print(f"  Reward: {d['reward']}, Done: {d['done']}")

# Test 5: Step - buy
print("\n=== /step (buy) ===")
r = requests.post(f"{BASE}/step", json={"action_type": "buy", "item_ids": [pids[0]]})
d = r.json()
print(f"  Reward: {d['reward']}, Done: {d['done']}")
print(f"  Cumulative: {d['info']['cumulative_reward']}")

# Test 6: State
print("\n=== /state ===")
r = requests.get(f"{BASE}/state")
d = r.json()
print(f"  Task: {d['task_name']}, Steps: {d['step_count']}, Done: {d['done']}")
print(f"  Cumulative reward: {d['cumulative_reward']}")

# Test 7: Reset with medium task
print("\n=== /reset (smart_shop) ===")
r = requests.post(f"{BASE}/reset", json={"task": "smart_shop"})
data = r.json()
info = data["info"]
print(f"  Task: {info['task']}, Products: {info['product_count']}, Max steps: {info['max_steps']}")

# Test 8: Reset with hard task
print("\n=== /reset (expert_deal) ===")
r = requests.post(f"{BASE}/reset", json={"task": "expert_deal"})
data = r.json()
info = data["info"]
print(f"  Task: {info['task']}, Products: {info['product_count']}, Max steps: {info['max_steps']}")

print("\n" + "=" * 50)
print("ALL OPENENV ENDPOINTS WORKING ✓")
print("=" * 50)
