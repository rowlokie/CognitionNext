#!/usr/bin/env python3
"""
Phase 4: Latency benchmarking
Measures p50/p95/p99 latency for each decision type.
Writes results to results.json.
Run against a running server: python scripts/bench.py
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
N_SAMPLES = 500  # per decision type
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_PATH = ROOT / "results.json"


def load_dev_samples(n=N_SAMPLES):
    dev_path = DATA_DIR / "dev.jsonl"
    samples = []
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    continue
    # Repeat if needed
    while len(samples) < n:
        samples.extend(samples)
    return samples[:n]


def percentile(data, p):
    if not data:
        return 0.0
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def warmup(session, n=20):
    dummy = {
        "request_id": "warmup",
        "decision": "route",
        "state": {
            "subject": "warmup",
            "body": "warmup body for latency test",
            "customer_tier": "pro",
            "amount": 50.0,
            "draft_reply": "warmup reply",
            "kb_article": "warmup article",
        }
    }
    for _ in range(n):
        session.post(f"{BASE_URL}/v1/decide", json=dummy, timeout=5)
    print(f"  Warmed up with {n} requests")


def bench_task(session, samples, decision, n=N_SAMPLES):
    latencies = []
    errors = 0
    for i, sample in enumerate(samples[:n]):
        payload = {
            "request_id": sample.get("request_id", f"bench-{i}"),
            "decision": decision,
            "state": sample.get("state", {}),
        }
        t0 = time.perf_counter()
        try:
            r = session.post(f"{BASE_URL}/v1/decide", json=payload, timeout=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1
        except Exception as e:
            errors += 1

    return latencies, errors


def run_bench():
    print("=" * 60)
    print("  LATENCY BENCHMARK (warm, sequential, single request)")
    print("=" * 60)

    session = requests.Session()
    # Check server
    try:
        r = session.get(f"{BASE_URL}/health", timeout=2)
        if r.status_code != 200:
            print(f"Server not healthy: {r.status_code}")
            return None
        print(f"  Server: {BASE_URL} - OK")
    except Exception as e:
        print(f"  Server not reachable: {e}")
        return None

    samples = load_dev_samples(N_SAMPLES)
    print(f"  Samples: {len(samples)} per decision type")

    print("\n  Warming up...")
    warmup(session, 30)

    results = {}
    all_pass = True

    for decision in ["route", "approve", "grounded"]:
        print(f"\n  Benchmarking '{decision}' ({N_SAMPLES} requests)...")
        latencies, errors = bench_task(session, samples, decision, N_SAMPLES)

        if not latencies:
            print(f"  [FAIL] No successful requests for {decision}")
            all_pass = False
            continue

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        p_mean = sum(latencies) / len(latencies)
        p_min = min(latencies)
        p_max = max(latencies)

        target_status = "[PASS] <=25ms" if p95 <= 25 else ("[PASS] <=100ms" if p95 <= 100 else "[FAIL] OVER BUDGET")

        print(f"  {decision}:")
        print(f"    p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
        print(f"    mean={p_mean:.2f}ms  min={p_min:.2f}ms  max={p_max:.2f}ms")
        print(f"    errors={errors}/{N_SAMPLES}  budget={target_status}")

        if p95 > 100:
            all_pass = False

        results[decision] = {
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "mean_ms": round(p_mean, 3),
            "min_ms": round(p_min, 3),
            "max_ms": round(p_max, 3),
            "n_requests": len(latencies),
            "n_errors": errors,
        }

    # Write to results.json
    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = {}

    existing["latency"] = results
    existing["latency"]["benchmark_config"] = {
        "n_samples": N_SAMPLES,
        "mode": "warm_sequential_single",
        "server": BASE_URL,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n  Results written to {RESULTS_PATH}")

    print("\n" + "=" * 60)
    if all_pass:
        print("  [PASS] All tasks within p95 <= 100ms budget")
    else:
        print("  [FAIL] Some tasks exceed latency budget")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_bench()
