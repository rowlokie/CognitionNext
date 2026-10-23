"""
Reproduce the three failure cases from the spec, then record baseline predictions.
Run against the live API and also directly through the model.
"""
import sys, json, pickle
sys.path.insert(0, 'src')
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/decide"
ROOT = Path(".")

KB_WARRANTY = (
    "Standard and Pro warranties strictly exclude liquid ingress, submersion, "
    "and cosmetic damage. Accidental liquid damage requires an optional Care+ policy."
)

CASES = [
    {
        "id": "test-A",
        "label": "UNGROUNDED",
        "desc": "Direct policy contradiction — warranty covers liquid immersion",
        "state": {
            "subject": "Water damage warranty claim",
            "body": "My phone fell in a swimming pool.",
            "customer_tier": "pro",
            "amount": 0.0,
            "draft_reply": "Your Pro warranty covers water damage caused by full immersion in a swimming pool.",
            "kb_article": KB_WARRANTY,
        }
    },
    {
        "id": "test-B",
        "label": "UNGROUNDED",
        "desc": "Unsupported deductible claim — Care+ coverage and zero deductible not in article",
        "state": {
            "subject": "Water damage warranty claim",
            "body": "My phone fell in a swimming pool.",
            "customer_tier": "pro",
            "amount": 0.0,
            "draft_reply": "Your Care+ policy covers the damage with a zero deductible.",
            "kb_article": KB_WARRANTY,
        }
    },
    {
        "id": "test-C",
        "label": "GROUNDED",
        "desc": "Supported reply — accurately reflects the article",
        "state": {
            "subject": "Water damage warranty claim",
            "body": "My phone fell in a swimming pool.",
            "customer_tier": "pro",
            "amount": 0.0,
            "draft_reply": "Liquid immersion and submersion are excluded from Standard and Pro warranties. Accidental liquid damage requires an optional Care+ policy.",
            "kb_article": KB_WARRANTY,
        }
    },
]

import requests

print("=" * 75)
print("  BASELINE FAILURE CASE REPRODUCTION (Via Live API)")
print("=" * 75)

for case in CASES:
    payload = {
        "request_id": case["id"],
        "decision": "grounded",
        "state": case["state"],
    }
    resp = requests.post(URL, json=payload, timeout=5)
    data = resp.json()
    p_true = data.get("p_true", -1)
    value = data.get("value")
    abstain = data.get("abstain")
    lat = data.get("latency_ms")
    expected = case["label"]
    predicted = "GROUNDED" if value else "UNGROUNDED"
    result = "[CORRECT]" if predicted == expected else "[WRONG] <<<< FAILURE"

    print(f"\nCase {case['id']}: {case['desc']}")
    print(f"  Expected:        {expected}")
    print(f"  Predicted:       {predicted}  {result}")
    print(f"  p_true:          {p_true:.6f}")
    print(f"  confidence:      {max(p_true, 1-p_true):.6f}")
    print(f"  abstain:         {abstain}")
    print(f"  latency_ms:      {lat:.2f}")
    print(f"  KB excerpt:      \"{case['state']['kb_article'][:80]}...\"")
    print(f"  Reply excerpt:   \"{case['state']['draft_reply'][:80]}\"")

print("\n" + "=" * 75)
print("  Done. Record these before making any changes.")
print("=" * 75)
