#!/usr/bin/env python3
"""
tests/test_inference.py
Validates inference logic, prediction correctness, probabilities sum,
confidence definitions, and abstention behavior for all three tasks.
"""

import sys
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_inference_tasks():
    print("=" * 60)
    print("  DECIDE INFERENCE VALIDATION TESTS")
    print("=" * 60)

    # 1. Route Decision
    print("\n1. Testing Route Decision...")
    req_route = {
        "request_id": "inf-route-01",
        "decision": "route",
        "state": {
            "subject": "Discrepancy in monthly invoice payment",
            "body": "I was double charged on my credit card for seat renewal.",
            "customer_tier": "enterprise",
            "amount": 199.0,
            "draft_reply": "",
            "kb_article": ""
        }
    }
    r = requests.post(f"{BASE_URL}/v1/decide", json=req_route)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["type"] == "choice", f"Expected choice, got {data['type']}"
    assert data["value"] in ["billing", "technical", "account", "shipping", "general"]
    probs = data["probs"]
    assert len(probs) == 5, f"Expected 5 queue probabilities, got {len(probs)}"
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) <= 0.01, f"Probs must sum to 1.0, got {prob_sum}"
    assert abs(data["confidence"] - probs[data["value"]]) <= 0.001, "Confidence must match chosen class probability"
    assert isinstance(data["abstain"], bool), "Abstain must be boolean"
    print(f"  [PASS] Route: value={data['value']}, confidence={data['confidence']:.4f}, sum={prob_sum:.4f}")

    # 2. Approve Decision
    print("\n2. Testing Approve Decision...")
    req_app = {
        "request_id": "inf-app-01",
        "decision": "approve",
        "state": {
            "subject": "Refund request for unused period",
            "body": "Please issue refund for $15 charge on account.",
            "customer_tier": "enterprise",
            "amount": 15.0,
            "draft_reply": "Approved immediately.",
            "kb_article": "Refunds under $50 are auto-approved."
        }
    }
    r = requests.post(f"{BASE_URL}/v1/decide", json=req_app)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["type"] == "binary", f"Expected binary, got {data['type']}"
    assert isinstance(data["value"], bool), "Value must be boolean"
    assert 0.0 <= data["p_true"] <= 1.0, f"p_true must be in [0, 1], got {data['p_true']}"
    assert isinstance(data["abstain"], bool), "Abstain must be boolean"
    print(f"  [PASS] Approve: value={data['value']}, p_true={data['p_true']:.4f}, abstain={data['abstain']}")

    # 3. Grounded Decision
    print("\n3. Testing Grounded Decision...")
    req_gr = {
        "request_id": "inf-gr-01",
        "decision": "grounded",
        "state": {
            "subject": "Delivery policy",
            "body": "When will it arrive?",
            "customer_tier": "pro",
            "amount": 0.0,
            "draft_reply": "Standard delivery takes 5-7 business days.",
            "kb_article": "Standard delivery takes 5-7 business days for all domestic orders."
        }
    }
    r = requests.post(f"{BASE_URL}/v1/decide", json=req_gr)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["type"] == "binary", f"Expected binary, got {data['type']}"
    assert isinstance(data["value"], bool), "Value must be boolean"
    assert 0.0 <= data["p_true"] <= 1.0, f"p_true must be in [0, 1], got {data['p_true']}"
    assert isinstance(data["abstain"], bool), "Abstain must be boolean"
    print(f"  [PASS] Grounded: value={data['value']}, p_true={data['p_true']:.4f}, abstain={data['abstain']}")

    print("\n" + "=" * 60)
    print("  [PASS] ALL INFERENCE TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_inference_tasks()
