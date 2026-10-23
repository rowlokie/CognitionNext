"""
Warm sequential latency benchmark — promoted grounded v2 model.
Measures p50 and p95 per decision type, 200 warm requests each.
"""
import sys, json, time, statistics, requests
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/decide"
ROOT = Path(".")

PAYLOADS = {
    "route": {
        "request_id": "bench-route",
        "decision": "route",
        "state": {"subject": "Billing issue", "body": "I was charged twice this month.", "customer_tier": "pro", "amount": 99.0},
    },
    "approve": {
        "request_id": "bench-approve",
        "decision": "approve",
        "state": {"subject": "Refund request", "body": "Please refund my order.", "customer_tier": "enterprise", "amount": 45.0},
    },
    "grounded": {
        "request_id": "bench-grounded",
        "decision": "grounded",
        "state": {
            "subject": "Warranty question",
            "body": "Is liquid damage covered?",
            "customer_tier": "pro",
            "amount": 0.0,
            "draft_reply": "Liquid immersion is excluded from Standard and Pro warranties. Care+ covers accidental liquid damage.",
            "kb_article": "Standard and Pro warranties strictly exclude liquid ingress, submersion, and cosmetic damage. Accidental liquid damage requires an optional Care+ policy.",
        },
    },
}

N_WARMUP = 20
N_BENCH  = 200

print("=" * 60)
print("  LATENCY BENCHMARK — promoted grounded v2")
print("=" * 60)

results = {}
for task, payload in PAYLOADS.items():
    # Warmup
    for _ in range(N_WARMUP):
        requests.post(URL, json=payload, timeout=5)
    # Benchmark
    lats = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        r = requests.post(URL, json=payload, timeout=5)
        lats.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(lats)
    p95 = sorted(lats)[int(0.95 * N_BENCH)]
    p99 = sorted(lats)[int(0.99 * N_BENCH)]
    results[task] = {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2)}
    flag = "OK" if p95 < 25 else "OVER BUDGET"
    print(f"  {task:<12} p50={p50:6.2f}ms  p95={p95:6.2f}ms  p99={p99:6.2f}ms  [{flag}]")

print("=" * 60)

# Update results.json
rp = ROOT / "results.json"
existing = json.loads(rp.read_text()) if rp.exists() else {}
existing.update({
    "latency_p50": results,
    "latency_p95": {k: v["p95"] for k, v in results.items()},
    "latency_p99": {k: v["p99"] for k, v in results.items()},
    "model_version": "grounded_v2_enriched_features",
})
rp.write_text(json.dumps(existing, indent=2))
print(f"\nresults.json updated.\n")
