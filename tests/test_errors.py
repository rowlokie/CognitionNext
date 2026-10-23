#!/usr/bin/env python3
"""
tests/test_errors.py
Validates edge cases, input sanitization, error responses,
and schema validation error handlers.
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

def test_error_handling():
    print("=" * 60)
    print("  DECIDE ERROR & EDGE-CASE HANDLING TESTS")
    print("=" * 60)

    # 1. Unknown decision type -> 422
    print("\n1. Testing unknown decision type...")
    r = requests.post(f"{BASE_URL}/v1/decide", json={
        "request_id": "err-01",
        "decision": "escalate",
        "state": {}
    })
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"  [PASS] Unknown decision correctly rejected with 422")

    # 2. Missing decision field -> 422
    print("\n2. Testing missing decision field...")
    r = requests.post(f"{BASE_URL}/v1/decide", json={
        "request_id": "err-02",
        "state": {}
    })
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"  [PASS] Missing decision correctly rejected with 422")

    # 3. Completely empty payload -> 422
    print("\n3. Testing empty payload...")
    r = requests.post(f"{BASE_URL}/v1/decide", json={})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"  [PASS] Empty payload correctly rejected with 422")

    # 4. Null state fields -> 200 (graceful coercion)
    print("\n4. Testing null state fields...")
    r = requests.post(f"{BASE_URL}/v1/decide", json={
        "request_id": "err-04",
        "decision": "route",
        "state": {
            "subject": None,
            "body": None,
            "customer_tier": None,
            "amount": None,
            "draft_reply": None,
            "kb_article": None
        }
    })
    assert r.status_code == 200, f"Expected 200 with null values coerced, got {r.status_code}"
    res = r.json()
    assert res["value"] in ["billing", "technical", "account", "shipping", "general"]
    print(f"  [PASS] Null state values gracefully coerced, response valid: value={res['value']}")

    # 5. Invalid amount format (string or negative) -> 200 (coerced safely)
    print("\n5. Testing invalid amount type...")
    r = requests.post(f"{BASE_URL}/v1/decide", json={
        "request_id": "err-05",
        "decision": "approve",
        "state": {
            "subject": "Test",
            "body": "Test",
            "amount": "not_a_number"
        }
    })
    assert r.status_code == 200, f"Expected 200 with invalid amount coerced, got {r.status_code}"
    print(f"  [PASS] Invalid amount coerced safely without crashing")

    print("\n" + "=" * 60)
    print("  [PASS] ALL ERROR & EDGE-CASE TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_error_handling()
