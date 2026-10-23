import requests
import json

URL = "http://127.0.0.1:8000/v1/decide"

test_cases = [
    {
        "title": "1. ROUTE DECISION — Billing Dispute (Double Charged)",
        "payload": {
            "request_id": "req-route-001",
            "decision": "route",
            "state": {
                "subject": "Double charged on my Visa card for monthly subscription",
                "body": "Hi, I checked my bank statement today and I see two separate charges of $49.99 on October 20th. Please refund the duplicate transaction immediately.",
                "customer_tier": "pro",
                "amount": 49.99,
                "draft_reply": "We apologize for the inconvenience. Let me review your transaction history.",
                "kb_article": "Duplicate charges are automatically flagged and refunded within 3-5 business days upon verification."
            }
        }
    },
    {
        "title": "2. ROUTE DECISION — Technical Issue (Server 500 error / API crash)",
        "payload": {
            "request_id": "req-route-002",
            "decision": "route",
            "state": {
                "subject": "API 500 internal server error during database sync",
                "body": "Our webhook integration is failing with HTTP 500 status codes since 14:00 UTC. The stack trace indicates a timeout in PostgreSQL connection pool.",
                "customer_tier": "enterprise",
                "amount": 0.0,
                "draft_reply": "We are investigating the connection pool latency.",
                "kb_article": "Postgres connection pools have an automatic failover mechanism."
            }
        }
    },
    {
        "title": "3. APPROVE DECISION — Enterprise Refund Under SLA Policy",
        "payload": {
            "request_id": "req-approve-003",
            "decision": "approve",
            "state": {
                "subject": "Service downtime compensation request",
                "body": "Our team experienced a 45-minute outage yesterday during our sprint demo. As an enterprise tier customer, we request the standard $100 service credit.",
                "customer_tier": "enterprise",
                "amount": 100.0,
                "draft_reply": "We have credited your account with $100 as per SLA terms.",
                "kb_article": "Enterprise SLA guarantees 99.9% uptime. Outages exceeding 30 minutes qualify for service credits up to $250."
            }
        }
    },
    {
        "title": "4. APPROVE DECISION — Free Tier High Amount (Should Auto-Hold / Not Approve)",
        "payload": {
            "request_id": "req-approve-004",
            "decision": "approve",
            "state": {
                "subject": "Refund request for $850 hardware",
                "body": "I want an instant refund of $850 for the equipment I purchased yesterday.",
                "customer_tier": "free",
                "amount": 850.0,
                "draft_reply": "Your request has been forwarded to tier 2 for manual review.",
                "kb_article": "Amounts exceeding $200 require manager approval and manual invoice verification."
            }
        }
    },
    {
        "title": "5. GROUNDED DECISION — Accurately Supported Reply",
        "payload": {
            "request_id": "req-grounded-005",
            "decision": "grounded",
            "state": {
                "subject": "Return policy window question",
                "body": "Can I still return an opened hardware item after 20 days?",
                "customer_tier": "free",
                "amount": 0.0,
                "draft_reply": "Yes, you can return opened hardware items within 30 days of delivery with original packaging and proof of purchase.",
                "kb_article": "Standard hardware return policy allows returns within 30 days of delivery. Items must include original packaging and receipt."
            }
        }
    },
    {
        "title": "6. GROUNDED DECISION — Hallucinated / Unsupported Reply",
        "payload": {
            "request_id": "req-grounded-006",
            "decision": "grounded",
            "state": {
                "subject": "Warranty coverage for water damage",
                "body": "Does my warranty cover water damage if dropped in a pool?",
                "customer_tier": "pro",
                "amount": 0.0,
                "draft_reply": "Yes, our pro warranty completely covers liquid immersion and accidental drops with zero deductible.",
                "kb_article": "Standard and Pro warranties strictly exclude liquid ingress, submersion, and cosmetic damage. Accidental liquid damage requires an optional Care+ policy."
            }
        }
    }
]

def main():
    print("=" * 80)
    print("  LIVE DECIDE ENGINE PREDICTION TEST (Sending Requests to /v1/decide)")
    print("=" * 80)

    for tc in test_cases:
        p = tc["payload"]
        print(f"\n>>> {tc['title']}")
        print(f"    Request ID:  {p['request_id']}")
        print(f"    Decision:    {p['decision']}")
        print(f"    State Input: subject='{p['state']['subject']}', tier={p['state']['customer_tier']}, amount=${p['state']['amount']}")
        
        try:
            resp = requests.post(URL, json=p, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"    <- HTTP {resp.status_code} ({data.get('latency_ms', 0):.2f} ms)")
                print("    Response JSON:")
                print("    " + json.dumps(data, indent=6).replace("\n", "\n    "))
            else:
                print(f"    <- HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"    <- Error connecting to {URL}: {e}")

    print("\n" + "=" * 80)
    print("  TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
