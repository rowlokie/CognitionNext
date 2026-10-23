#!/usr/bin/env python3
"""
selfcheck.py — validates API response schema for all three decision types.
Usage: python selfcheck.py [--url http://localhost:8000] [--file examples.json]
"""

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

ROUTE_CLASSES = {"billing", "technical", "account", "shipping", "general"}
REQUIRED_ROUTE = {"request_id", "decision", "type", "value", "probs", "confidence", "abstain", "latency_ms"}
REQUIRED_BINARY = {"request_id", "decision", "type", "value", "p_true", "abstain", "latency_ms"}


def check_route(resp: dict, request_id: str) -> list:
    errors = []
    missing = REQUIRED_ROUTE - set(resp.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")
    if resp.get("type") != "choice":
        errors.append(f"type={resp.get('type')!r}, expected 'choice'")
    if resp.get("value") not in ROUTE_CLASSES:
        errors.append(f"value={resp.get('value')!r} not in route classes")
    probs = resp.get("probs", {})
    if set(probs.keys()) != ROUTE_CLASSES:
        errors.append(f"probs keys {set(probs.keys())} != {ROUTE_CLASSES}")
    prob_sum = sum(probs.values())
    if abs(prob_sum - 1.0) > 0.01:
        errors.append(f"probs sum={prob_sum:.4f}, not within 0.01 of 1.0")
    if not isinstance(resp.get("abstain"), bool):
        errors.append(f"abstain is {type(resp.get('abstain'))}, expected bool")
    if not isinstance(resp.get("latency_ms"), (int, float)):
        errors.append(f"latency_ms is {type(resp.get('latency_ms'))}, expected float")
    conf = resp.get("confidence")
    expected = probs.get(resp.get("value", ""), None)
    if conf is not None and expected is not None and abs(conf - expected) > 0.01:
        errors.append(f"confidence={conf} != probs[value]={expected}")
    return errors


def check_binary(resp: dict, decision: str) -> list:
    errors = []
    missing = REQUIRED_BINARY - set(resp.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")
    if resp.get("type") != "binary":
        errors.append(f"type={resp.get('type')!r}, expected 'binary'")
    if not isinstance(resp.get("value"), bool):
        errors.append(f"value is {type(resp.get('value'))}, expected bool")
    p = resp.get("p_true")
    if p is not None and not (0.0 <= p <= 1.0):
        errors.append(f"p_true={p} not in [0.0, 1.0]")
    if not isinstance(resp.get("abstain"), bool):
        errors.append(f"abstain is {type(resp.get('abstain'))}, expected bool")
    if not isinstance(resp.get("latency_ms"), (int, float)):
        errors.append(f"latency_ms is {type(resp.get('latency_ms'))}, expected float")
    return errors


def validate_response(resp: dict) -> list:
    """Validate a single response dict. Returns list of errors (empty = valid)."""
    decision = resp.get("decision")
    request_id = str(resp.get("request_id", ""))

    if decision == "route":
        return check_route(resp, request_id)
    elif decision in ("approve", "grounded"):
        return check_binary(resp, decision)
    else:
        return [f"Unknown decision: {decision!r}"]


def selfcheck_live(url: str, n_examples: int = 5) -> bool:
    """Run selfcheck against live server with sample requests."""
    import requests

    test_cases = [
        {
            "request_id": "sc-001",
            "decision": "route",
            "state": {
                "subject": "billing issue",
                "body": "I was charged twice this month",
                "customer_tier": "pro",
                "amount": 99.0,
                "draft_reply": "We'll look into this immediately.",
                "kb_article": "Billing disputes resolved within 24h.",
            }
        },
        {
            "request_id": "sc-002",
            "decision": "approve",
            "state": {
                "subject": "refund request",
                "body": "Please refund my last payment",
                "customer_tier": "enterprise",
                "amount": 50.0,
                "draft_reply": "Approved.",
                "kb_article": "Refunds within 30 days.",
            }
        },
        {
            "request_id": "sc-003",
            "decision": "grounded",
            "state": {
                "subject": "tracking",
                "body": "Where is my order?",
                "customer_tier": "free",
                "amount": 25.0,
                "draft_reply": "Standard delivery takes 5-7 days.",
                "kb_article": "Standard delivery takes 5-7 business days.",
            }
        },
    ]

    all_ok = True
    print(f"SelfCheck: {url}/v1/decide")
    print("=" * 50)
    for tc in test_cases:
        try:
            r = requests.post(f"{url}/v1/decide", json=tc, timeout=5)
            if r.status_code != 200:
                print(f"  [{tc['decision']}] HTTP {r.status_code}: {r.text[:200]}")
                all_ok = False
                continue
            resp = r.json()
            errors = validate_response(resp)
            if errors:
                print(f"  [{tc['decision']}] SCHEMA ERRORS: {errors}")
                all_ok = False
            else:
                val = resp.get("value")
                print(f"  [{tc['decision']}] OK - value={val}")
        except Exception as e:
            print(f"  [{tc['decision']}] ERROR: {e}")
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--file", help="JSON file with response objects to validate")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        all_ok = True
        for item in data:
            if "request" in item:
                import requests
                r = requests.post(f"{args.url}/v1/decide", json=item["request"], timeout=5)
                resp = r.json()
            else:
                resp = item
            errs = validate_response(resp)
            if errs:
                print(f"INVALID: {errs} - {resp.get('request_id')}")
                all_ok = False
            else:
                print(f"OK: {resp.get('request_id')} [{resp.get('decision')}] -> value={resp.get('value')} (abstain={resp.get('abstain')})")
        sys.exit(0 if all_ok else 1)
    else:
        ok = selfcheck_live(args.url)
        sys.exit(0 if ok else 1)
