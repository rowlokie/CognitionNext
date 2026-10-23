#!/usr/bin/env python3
"""
Schema and API tests for the Decide endpoint.
Tests:
1. Schema validity for all three decision types
2. Unknown decision type returns 422
3. Null/missing fields handled gracefully
4. Probs sum to 1 within 0.01 for route
5. p_true in [0,1] for binary tasks
6. All required fields present
"""

import json
import sys
import time
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"


def post_decide(payload: dict, timeout: float = 5.0):
    r = requests.post(f"{BASE_URL}/v1/decide", json=payload, timeout=timeout)
    return r.status_code, r.json() if r.content else {}


def check_route_schema(resp: dict, request_id: str):
    errors = []
    required = ["request_id", "decision", "type", "value", "probs", "confidence", "abstain", "latency_ms"]
    for f in required:
        if f not in resp:
            errors.append(f"Missing field: {f}")
    if "type" in resp and resp["type"] != "choice":
        errors.append(f"type should be 'choice', got {resp['type']!r}")
    if "value" in resp:
        allowed = {"billing", "technical", "account", "shipping", "general"}
        if resp["value"] not in allowed:
            errors.append(f"value {resp['value']!r} not in allowed route classes")
    if "probs" in resp:
        probs = resp["probs"]
        required_keys = {"billing", "technical", "account", "shipping", "general"}
        missing_keys = required_keys - set(probs.keys())
        if missing_keys:
            errors.append(f"probs missing keys: {missing_keys}")
        prob_sum = sum(probs.values())
        if abs(prob_sum - 1.0) > 0.01:
            errors.append(f"probs sum = {prob_sum:.4f}, not within 0.01 of 1.0")
    if "confidence" in resp and "probs" in resp and "value" in resp:
        expected_conf = resp["probs"].get(resp["value"], 0)
        if abs(resp["confidence"] - expected_conf) > 0.01:
            errors.append(f"confidence {resp['confidence']} != probs[value] {expected_conf}")
    if "abstain" in resp and not isinstance(resp["abstain"], bool):
        errors.append(f"abstain should be bool, got {type(resp['abstain'])}")
    if "latency_ms" in resp and not isinstance(resp["latency_ms"], (int, float)):
        errors.append(f"latency_ms should be float, got {type(resp['latency_ms'])}")
    if "request_id" in resp and resp["request_id"] != request_id:
        errors.append(f"request_id mismatch: {resp['request_id']!r} != {request_id!r}")
    return errors


def check_binary_schema(resp: dict, decision: str, request_id: str):
    errors = []
    required = ["request_id", "decision", "type", "value", "p_true", "abstain", "latency_ms"]
    for f in required:
        if f not in resp:
            errors.append(f"Missing field: {f}")
    if "type" in resp and resp["type"] != "binary":
        errors.append(f"type should be 'binary', got {resp['type']!r}")
    if "value" in resp and not isinstance(resp["value"], bool):
        errors.append(f"value should be bool, got {type(resp['value'])}")
    if "p_true" in resp:
        p = resp["p_true"]
        if not (0.0 <= p <= 1.0):
            errors.append(f"p_true {p} not in [0, 1]")
    if "abstain" in resp and not isinstance(resp["abstain"], bool):
        errors.append(f"abstain should be bool, got {type(resp['abstain'])}")
    if "request_id" in resp and resp["request_id"] != request_id:
        errors.append(f"request_id mismatch: {resp['request_id']!r} != {request_id!r}")
    return errors


def run_tests():
    print("=" * 60)
    print("  DECIDE API SCHEMA TESTS")
    print("=" * 60)

    # Wait for server
    for attempt in range(10):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"  [OK] Server is up")
                break
        except Exception:
            print(f"  Waiting for server... ({attempt+1}/10)")
            time.sleep(1)
    else:
        print("  [FAIL] Server not responding. Start with: python -m uvicorn app.main:app")
        return False

    all_passed = True

    base_state = {
        "subject": "Need help with my order",
        "body": "My order hasn't arrived and it's been 2 weeks.",
        "customer_tier": "pro",
        "amount": 45.99,
        "draft_reply": "We apologize for the delay. Your order will arrive within 5 business days.",
        "kb_article": "Standard delivery takes 5-7 business days. Contact support for delays over 10 days.",
    }

    # Test 1: Route decision schema
    print("\n  Test 1: Route decision schema")
    payload = {"request_id": "test-001", "decision": "route", "state": base_state}
    status, resp = post_decide(payload)
    errors = []
    if status != 200:
        errors.append(f"HTTP {status}: {resp}")
    else:
        errors = check_route_schema(resp, "test-001")
    if errors:
        print(f"  [FAIL]: {errors}")
        all_passed = False
    else:
        print(f"  [PASS] (value={resp.get('value')}, confidence={resp.get('confidence'):.3f})")

    # Test 2: Approve decision schema
    print("\n  Test 2: Approve decision schema")
    payload = {"request_id": "test-002", "decision": "approve", "state": base_state}
    status, resp = post_decide(payload)
    errors = []
    if status != 200:
        errors.append(f"HTTP {status}: {resp}")
    else:
        errors = check_binary_schema(resp, "approve", "test-002")
    if errors:
        print(f"  [FAIL]: {errors}")
        all_passed = False
    else:
        print(f"  [PASS] (value={resp.get('value')}, p_true={resp.get('p_true'):.3f})")

    # Test 3: Grounded decision schema
    print("\n  Test 3: Grounded decision schema")
    payload = {"request_id": "test-003", "decision": "grounded", "state": base_state}
    status, resp = post_decide(payload)
    errors = []
    if status != 200:
        errors.append(f"HTTP {status}: {resp}")
    else:
        errors = check_binary_schema(resp, "grounded", "test-003")
    if errors:
        print(f"  [FAIL]: {errors}")
        all_passed = False
    else:
        print(f"  [PASS] (value={resp.get('value')}, p_true={resp.get('p_true'):.3f})")

    # Test 4: Unknown decision -> 422
    print("\n  Test 4: Unknown decision type -> 422")
    payload = {"request_id": "test-004", "decision": "unknown", "state": base_state}
    status, resp = post_decide(payload)
    if status == 422:
        print(f"  [PASS] (HTTP 422 as expected)")
    else:
        print(f"  [FAIL]: Expected 422, got {status}")
        all_passed = False

    # Test 5: Null fields handled gracefully
    print("\n  Test 5: Null fields handled gracefully")
    null_state = {
        "subject": None,
        "body": None,
        "customer_tier": None,
        "amount": None,
        "draft_reply": None,
        "kb_article": None,
    }
    for decision in ["route", "approve", "grounded"]:
        payload = {"request_id": f"test-005-{decision}", "decision": decision, "state": null_state}
        status, resp = post_decide(payload)
        if status != 200:
            print(f"  [FAIL] for {decision}: HTTP {status}: {resp}")
            all_passed = False
        else:
            print(f"  [PASS] {decision}: handled null fields (HTTP 200)")

    # Test 6: Missing state fields (partial state)
    print("\n  Test 6: Partial state (missing fields)")
    partial_state = {"subject": "Test"}
    payload = {"request_id": "test-006", "decision": "route", "state": partial_state}
    status, resp = post_decide(payload)
    if status == 200:
        errors = check_route_schema(resp, "test-006")
        if errors:
            print(f"  [FAIL]: {errors}")
            all_passed = False
        else:
            print(f"  [PASS]: partial state handled, schema valid")
    else:
        print(f"  [FAIL]: HTTP {status}: {resp}")
        all_passed = False

    # Test 7: Latency check
    print("\n  Test 7: Single-request latency check (warm)")
    latencies = []
    for i in range(20):
        t0 = time.perf_counter()
        status, resp = post_decide(
            {"request_id": f"lat-{i}", "decision": "route", "state": base_state}
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if status == 200:
            latencies.append(elapsed)
    if latencies:
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        p50 = latencies[len(latencies) // 2]
        print(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  (target p95<=100ms)")
        if p95 <= 100:
            print(f"  [PASS] (within p95 budget)")
        else:
            print(f"  [FAIL]: p95={p95:.1f}ms exceeds 100ms")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  [PASS] ALL TESTS PASSED")
    else:
        print("  [FAIL] SOME TESTS FAILED")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
