import requests, json

URL = "http://127.0.0.1:8000/v1/decide"

TESTS = [
    {
        "id": "Test 1",
        "expected": False,
        "kb": "Standard and Pro warranties exclude liquid ingress and submersion.",
        "reply": "Pro warranty covers water damage caused by full immersion in a swimming pool."
    },
    {
        "id": "Test 2",
        "expected": False,
        "kb": "Care+ is optional; applicable charges are specified in the customer's policy.",
        "reply": "Care+ has a zero deductible."
    },
    {
        "id": "Test 7",
        "expected": False,
        "kb": "Returns allowed within 14 calendar days, subject to conditions and exclusions.",
        "reply": "Returns allowed within 30 days with no exceptions."
    },
    {
        "id": "Test 8",
        "expected": False,
        "kb": "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.",
        "reply": "Refund processed within 30 calendar days."
    },
    {
        "id": "Test 9",
        "expected": True,
        "kb": "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.",
        "reply": "Approved refunds are processed within 7 calendar days after approval; bank posting times may vary."
    },
    {
        "id": "Test 10",
        "expected": False,
        "kb": "Standard warranty lasts 12 months.",
        "reply": "Standard warranty lasts 24 months."
    },
    {
        "id": "Test 11",
        "expected": True,
        "kb": "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions.",
        "reply": "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions."
    },
    {
        "id": "Test 12",
        "expected": False,
        "kb": "Free replacements are limited to confirmed manufacturing defects; accidental and physical damage are excluded.",
        "reply": "Guarantees a free replacement for every damaged device regardless of cause."
    },
    {
        "id": "Test 13",
        "expected": False,
        "kb": "Replacement fee is $100.",
        "reply": "Replacement fee is $250."
    },
]

print(f"{'ID':<8} {'Expected':<10} {'Predicted':<10} {'p_true':<10} {'Abstain':<8} {'Status'}")
print("-" * 65)
for t in TESTS:
    payload = {
        "request_id": t["id"],
        "decision": "grounded",
        "state": {
            "customer_tier": "pro",
            "amount": 0.0,
            "draft_reply": t["reply"],
            "kb_article": t["kb"],
        }
    }
    r = requests.post(URL, json=payload).json()
    val = r["value"]
    p_true = r["p_true"]
    abstain = r["abstain"]
    status = "PASS" if val == t["expected"] else "FAIL"
    print(f"{t['id']:<8} {str(t['expected']):<10} {str(val):<10} {p_true:<10.4f} {str(abstain):<8} {status}")
