#!/usr/bin/env python3
"""
interactive_cli.py — Send custom requests to the Decide API.
Supports both interactive loop and CLI arguments.
"""
import sys
import json
import argparse
import requests

URL = "http://127.0.0.1:8000/v1/decide"

def send_request(decision, subject, body, tier="pro", amount=0.0, reply="", kb=""):
    payload = {
        "request_id": "manual-test",
        "decision": decision,
        "state": {
            "subject": subject,
            "body": body,
            "customer_tier": tier,
            "amount": float(amount),
            "draft_reply": reply,
            "kb_article": kb,
        }
    }
    print("\n" + "=" * 60)
    print("  REQUEST PAYLOAD")
    print("=" * 60)
    print(json.dumps(payload, indent=2))

    try:
        resp = requests.post(URL, json=payload, timeout=5)
        print("\n" + "=" * 60)
        print(f"  RESPONSE (HTTP {resp.status_code})")
        print("=" * 60)
        print(json.dumps(resp.json(), indent=2))
        return resp.json()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to {URL}: {e}")
        print("Make sure the server is running on port 8000.")
        return None

def interactive_mode():
    print("=" * 60)
    print("  Decide Engine — Interactive Request Tester")
    print("  Server URL: http://127.0.0.1:8000/v1/decide")
    print("=" * 60)

    while True:
        print("\n--- Enter request details (or press Enter for default, 'q' to quit) ---")
        dec = input("Choose decision [route / approve / grounded] (default: grounded): ").strip().lower()
        if dec in ("q", "quit", "exit"):
            break
        if not dec:
            dec = "grounded"

        if dec == "grounded":
            kb = input("KB Article (default: warranty policy): ").strip()
            if not kb:
                kb = "Standard and Pro warranties strictly exclude liquid ingress, submersion, and cosmetic damage. Accidental liquid damage requires an optional Care+ policy."
            reply = input("Draft Reply (default: policy contradiction): ").strip()
            if not reply:
                reply = "Your Pro warranty covers water damage caused by full immersion in a swimming pool."
            subj = "Warranty claim"
            body = "My device has water damage."
            tier = "pro"
            amt = 0.0
        else:
            subj = input("Subject: ").strip() or "Need assistance with payment"
            body = input("Body: ").strip() or "I was charged twice for subscription"
            tier = input("Tier [free / pro / enterprise] (default: pro): ").strip().lower() or "pro"
            amt_str = input("Amount (default: 50.0): ").strip()
            try:
                amt = float(amt_str) if amt_str else 50.0
            except ValueError:
                amt = 50.0
            reply = ""
            kb = ""

        send_request(dec, subj, body, tier, amt, reply, kb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Decide Engine API")
    parser.add_argument("--decision", choices=["route", "approve", "grounded"], help="Decision task")
    parser.add_argument("--subject", default="Test request")
    parser.add_argument("--body", default="Test body")
    parser.add_argument("--tier", default="pro", choices=["free", "pro", "enterprise"])
    parser.add_argument("--amount", type=float, default=0.0)
    parser.add_argument("--reply", default="")
    parser.add_argument("--kb", default="")

    args = parser.parse_args()

    if args.decision:
        send_request(
            decision=args.decision,
            subject=args.subject,
            body=args.body,
            tier=args.tier,
            amount=args.amount,
            reply=args.reply,
            kb=args.kb,
        )
    else:
        interactive_mode()
