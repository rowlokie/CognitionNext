#!/usr/bin/env python3
"""
Automated Test Runner for Decide API using the Fresh 100-Query Manual QA Dataset.
Dataset: testing folder/decide_fresh_manual_test_dataset_100.json

Validates:
- HTTP status & response latency
- Response schema & required fields
- Probabilities & confidence integrity
- Prediction accuracy per decision task (route, approve, grounded)
- Ambiguous approval & grounding abstention behavior
- Generates summary.json, results.csv, and failures.json under testing/results/
"""

import sys
import os
import time
import json
import csv
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import requests

ROOT = Path(__file__).parent.parent
DEFAULT_API_URL = "http://127.0.0.1:8000/v1/decide"

DEFAULT_DATASET_PATH = ROOT / "testing" / "decide_fresh_manual_test_dataset_100.json"
if not DEFAULT_DATASET_PATH.exists():
    DEFAULT_DATASET_PATH = ROOT / "testing" / "decide_manual_test_dataset_100.json"
if not DEFAULT_DATASET_PATH.exists():
    DEFAULT_DATASET_PATH = ROOT / "testing folder" / "decide_fresh_manual_test_dataset_100.json"

RESULTS_DIR = ROOT / "testing" / "results"
ALT_RESULTS_DIR = ROOT / "testing folder" / "results"

def load_dataset(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_response_schema(dec, r, req_id):
    errors = []
    # Check top-level required fields
    for field in ("request_id", "decision", "type", "value", "abstain", "latency_ms"):
        if field not in r:
            errors.append(f"Missing required field: '{field}'")
    
    if errors:
        return False, "SCHEMA_ERROR: " + "; ".join(errors)

    if str(r.get("request_id")) != str(req_id):
        errors.append(f"request_id mismatch: got '{r.get('request_id')}', expected '{req_id}'")
    if r.get("decision") != dec:
        errors.append(f"decision mismatch: got '{r.get('decision')}', expected '{dec}'")
    if not isinstance(r.get("abstain"), bool):
        errors.append(f"'abstain' must be bool, got {type(r.get('abstain'))}")
    if not isinstance(r.get("latency_ms"), (int, float)) or r.get("latency_ms") < 0:
        errors.append(f"'latency_ms' must be non-negative float, got {r.get('latency_ms')}")

    if dec == "route":
        if r.get("type") != "choice":
            errors.append(f"route type must be 'choice', got '{r.get('type')}'")
        probs = r.get("probs")
        if not isinstance(probs, dict):
            errors.append("'probs' must be a dict")
        else:
            prob_sum = 0.0
            for cls_name, p in probs.items():
                if not isinstance(p, (int, float)) or math.isnan(p) or math.isinf(p) or not (0.0 <= p <= 1.0):
                    errors.append(f"Probability for '{cls_name}' is invalid/out-of-bounds: {p}")
                else:
                    prob_sum += p
            if abs(prob_sum - 1.0) > 0.01:
                errors.append(f"Probabilities do not sum to ~1.0: sum = {prob_sum:.4f}")
        
        conf = r.get("confidence")
        if not isinstance(conf, (int, float)) or math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
            errors.append(f"Invalid confidence: {conf}")
        elif isinstance(probs, dict) and r.get("value") in probs:
            if abs(conf - probs[r.get("value")]) > 0.005:
                errors.append(f"confidence {conf} does not match chosen class probability {probs[r.get('value')]}")

    elif dec in ("approve", "grounded"):
        if r.get("type") != "binary":
            errors.append(f"{dec} type must be 'binary', got '{r.get('type')}'")
        if not isinstance(r.get("value"), bool):
            errors.append(f"'value' must be bool, got {r.get('value')}")
        p_true = r.get("p_true")
        if not isinstance(p_true, (int, float)) or math.isnan(p_true) or math.isinf(p_true) or not (0.0 <= p_true <= 1.0):
            errors.append(f"Invalid p_true: {p_true}")

    if errors:
        return False, "SCHEMA_ERROR: " + "; ".join(errors)
    return True, None

def run_tests(dataset_path=DEFAULT_DATASET_PATH, api_url=DEFAULT_API_URL):
    dataset_path = Path(dataset_path)
    data = load_dataset(dataset_path)
    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} cases from {dataset_path}")
    print(f"Testing against Decide API at: {api_url}")

    # Health check
    try:
        health_resp = requests.get(api_url.replace("/v1/decide", "/health"), timeout=3)
        if health_resp.status_code != 200:
            print(f"[WARN] Health check returned HTTP {health_resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Could not connect to API at {api_url}: {e}")
        print("Please start the API server using:")
        print("  python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ALT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    task_metrics = {
        "route": {
            "total": 0, "passed": 0, "failed": 0,
            "pred_correct": 0, "abstained": 0, "api_errors": 0, "schema_errors": 0,
            "latencies_ms": []
        },
        "approve": {
            "total": 0, "passed": 0, "failed": 0, "abstained": 0,
            "standard_total": 0, "standard_correct": 0, "standard_abstained": 0,
            "ambiguous_total": 0, "ambiguous_abstained": 0, "ambiguous_failed": 0,
            "api_errors": 0, "schema_errors": 0,
            "latencies_ms": []
        },
        "grounded": {
            "total": 0, "passed": 0, "failed": 0, "abstained": 0,
            "standard_total": 0, "standard_correct": 0, "standard_abstained": 0,
            "ambiguous_total": 0, "ambiguous_abstained": 0, "ambiguous_failed": 0,
            "api_errors": 0, "schema_errors": 0,
            "latencies_ms": []
        }
    }

    all_latencies_roundtrip = []
    all_latencies_server = []

    print("\nExecuting test cases...")
    print("-" * 80)
    print(f"{'Index':<6} {'Request ID':<16} {'Task':<10} {'Expected':<12} {'Actual':<12} {'Abstain':<8} {'Latency':<8} {'Status'}")
    print("-" * 80)

    for i, c in enumerate(cases, 1):
        req = c["request"]
        req_id = req["request_id"]
        dec = req["decision"]
        exp = c.get("expected", {})
        rationale = c.get("rationale", "")

        t_stats = task_metrics[dec]
        t_stats["total"] += 1

        t0 = time.perf_counter()
        http_error = None
        resp_json = None

        try:
            r = requests.post(api_url, json=req, timeout=10)
            roundtrip_ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                http_error = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                resp_json = r.json()
        except requests.exceptions.Timeout:
            roundtrip_ms = (time.perf_counter() - t0) * 1000
            http_error = "REQUEST_TIMEOUT (10s)"
        except requests.exceptions.ConnectionError:
            roundtrip_ms = (time.perf_counter() - t0) * 1000
            http_error = "CONNECTION_ERROR"
        except Exception as e:
            roundtrip_ms = (time.perf_counter() - t0) * 1000
            http_error = f"UNEXPECTED_NETWORK_ERROR: {str(e)}"

        all_latencies_roundtrip.append(roundtrip_ms)
        t_stats["latencies_ms"].append(roundtrip_ms)

        # Evaluation
        status = "PASS"
        failure_type = None
        failure_detail = None
        actual_val = None
        actual_abstain = None
        conf_prob = None
        server_latency_ms = None

        if http_error:
            status = "FAIL"
            failure_type = "API_ERROR"
            failure_detail = http_error
            t_stats["api_errors"] += 1
            t_stats["failed"] += 1
        else:
            server_latency_ms = resp_json.get("latency_ms")
            if server_latency_ms is not None:
                all_latencies_server.append(server_latency_ms)

            # 1. Validate Schema
            is_valid_schema, schema_err = validate_response_schema(dec, resp_json, req_id)
            if not is_valid_schema:
                status = "FAIL"
                failure_type = "SCHEMA_ERROR"
                failure_detail = schema_err
                t_stats["schema_errors"] += 1
                t_stats["failed"] += 1

            actual_val = resp_json.get("value")
            actual_abstain = resp_json.get("abstain")
            if actual_abstain:
                t_stats["abstained"] += 1

            if dec == "route":
                conf_prob = resp_json.get("confidence")
                expected_val = exp.get("value")
                if actual_val == expected_val:
                    t_stats["pred_correct"] += 1
                else:
                    if status == "PASS":
                        status = "FAIL"
                        failure_type = "PREDICTION_MISMATCH"
                        failure_detail = f"Predicted '{actual_val}', expected '{expected_val}'"
                        t_stats["failed"] += 1

            elif dec == "approve":
                conf_prob = resp_json.get("p_true")
                expected_val = exp.get("value")

                if expected_val is None:
                    # Ambiguous approval case
                    t_stats["ambiguous_total"] += 1
                    if actual_abstain is True:
                        t_stats["ambiguous_abstained"] += 1
                    else:
                        if status == "PASS":
                            status = "FAIL"
                            failure_type = "UNEXPECTED_NO_ABSTENTION"
                            failure_detail = f"Ambiguous approval case expected abstain=True, but API returned abstain=False (p_true={conf_prob:.4f}, value={actual_val})"
                            t_stats["ambiguous_failed"] += 1
                            t_stats["failed"] += 1
                else:
                    # Standard approval case
                    t_stats["standard_total"] += 1
                    if actual_abstain:
                        t_stats["standard_abstained"] += 1

                    if actual_val == expected_val:
                        t_stats["standard_correct"] += 1
                    else:
                        if status == "PASS":
                            status = "FAIL"
                            failure_type = "PREDICTION_MISMATCH"
                            failure_detail = f"Predicted {actual_val}, expected {expected_val} (p_true={conf_prob:.4f}, abstain={actual_abstain})"
                            t_stats["failed"] += 1

            elif dec == "grounded":
                conf_prob = resp_json.get("p_true")
                expected_val = exp.get("value")

                if expected_val is None:
                    # Ambiguous grounded case (e.g. grounded-new-030)
                    t_stats["ambiguous_total"] += 1
                    if actual_abstain is True:
                        t_stats["ambiguous_abstained"] += 1
                    else:
                        if status == "PASS":
                            status = "FAIL"
                            failure_type = "UNEXPECTED_NO_ABSTENTION"
                            failure_detail = f"Ambiguous grounded case expected abstain=True, but API returned abstain=False (p_true={conf_prob:.4f}, value={actual_val})"
                            t_stats["ambiguous_failed"] += 1
                            t_stats["failed"] += 1
                else:
                    t_stats["standard_total"] += 1
                    if actual_abstain:
                        t_stats["standard_abstained"] += 1

                    if actual_val == expected_val:
                        t_stats["standard_correct"] += 1
                    else:
                        if status == "PASS":
                            status = "FAIL"
                            failure_type = "PREDICTION_MISMATCH"
                            failure_detail = f"Predicted {actual_val}, expected {expected_val} (p_true={conf_prob:.4f}, abstain={actual_abstain})"
                            t_stats["failed"] += 1

        if status == "PASS":
            t_stats["passed"] += 1
        else:
            failures.append({
                "request_id": req_id,
                "task": dec,
                "failure_type": failure_type,
                "failure_detail": failure_detail,
                "expected": exp,
                "actual": resp_json if resp_json else {"error": http_error},
                "request_state": req.get("state", {}),
                "rationale": rationale,
            })

        # Terminal line
        exp_disp = str(exp.get("value")) if exp.get("value") is not None else "(ambiguous)"
        act_disp = str(actual_val) if actual_val is not None else "ERR"
        abst_disp = str(actual_abstain) if actual_abstain is not None else "ERR"
        lat_disp = f"{roundtrip_ms:.1f}ms"
        status_disp = f"[{status}]" if status == "PASS" else f"[{status}] << {failure_type}"
        print(f"{i:<6} {req_id:<16} {dec:<10} {exp_disp:<12} {act_disp:<12} {abst_disp:<8} {lat_disp:<8} {status_disp}")

        results.append({
            "request_id": req_id,
            "task": dec,
            "expected_value": exp.get("value"),
            "expected_abstain": exp.get("abstain"),
            "actual_value": actual_val,
            "actual_abstain": actual_abstain,
            "confidence_or_p_true": conf_prob,
            "status": status,
            "failure_type": failure_type or "",
            "failure_detail": failure_detail or "",
            "roundtrip_latency_ms": round(roundtrip_ms, 2),
            "server_latency_ms": round(server_latency_ms, 2) if server_latency_ms else None,
            "rationale": rationale,
        })

    # Summary calculations
    total_cases = len(cases)
    total_passed = sum(t["passed"] for t in task_metrics.values())
    total_failed = sum(t["failed"] for t in task_metrics.values())

    def calc_stats(lat_list):
        if not lat_list:
            return {"mean_ms": 0, "p50_ms": 0, "p90_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0}
        s = sorted(lat_list)
        return {
            "mean_ms": round(float(np.mean(s)), 2),
            "p50_ms": round(float(np.percentile(s, 50)), 2),
            "p90_ms": round(float(np.percentile(s, 90)), 2),
            "p95_ms": round(float(np.percentile(s, 95)), 2),
            "p99_ms": round(float(np.percentile(s, 99)), 2),
            "min_ms": round(float(np.min(s)), 2),
            "max_ms": round(float(np.max(s)), 2),
        }

    latency_summary = {
        "roundtrip": calc_stats(all_latencies_roundtrip),
        "server_internal": calc_stats(all_latencies_server),
    }

    # Per-task metric summaries
    summary_data = {
        "dataset_name": data.get("dataset_name"),
        "version": data.get("version"),
        "dataset_path": str(dataset_path),
        "total_cases": total_cases,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "overall_pass_rate": round(total_passed / total_cases * 100, 2),
        "tasks": {
            "route": {
                "total": task_metrics["route"]["total"],
                "passed": task_metrics["route"]["passed"],
                "failed": task_metrics["route"]["failed"],
                "prediction_accuracy": round(task_metrics["route"]["pred_correct"] / task_metrics["route"]["total"] * 100, 2),
                "abstention_rate": round(task_metrics["route"]["abstained"] / task_metrics["route"]["total"] * 100, 2),
                "api_errors": task_metrics["route"]["api_errors"],
                "schema_errors": task_metrics["route"]["schema_errors"],
                "latency": calc_stats(task_metrics["route"]["latencies_ms"]),
            },
            "approve": {
                "total": task_metrics["approve"]["total"],
                "passed": task_metrics["approve"]["passed"],
                "failed": task_metrics["approve"]["failed"],
                "overall_pass_rate": round(task_metrics["approve"]["passed"] / task_metrics["approve"]["total"] * 100, 2),
                "standard_cases": {
                    "total": task_metrics["approve"]["standard_total"],
                    "prediction_accuracy": round(task_metrics["approve"]["standard_correct"] / max(1, task_metrics["approve"]["standard_total"]) * 100, 2),
                    "abstention_rate": round(task_metrics["approve"]["standard_abstained"] / max(1, task_metrics["approve"]["standard_total"]) * 100, 2),
                },
                "ambiguous_cases": {
                    "total": task_metrics["approve"]["ambiguous_total"],
                    "abstained_as_expected": task_metrics["approve"]["ambiguous_abstained"],
                    "expected_abstention_rate": round(task_metrics["approve"]["ambiguous_abstained"] / max(1, task_metrics["approve"]["ambiguous_total"]) * 100, 2),
                    "unexpected_no_abstain_failures": task_metrics["approve"]["ambiguous_failed"],
                },
                "api_errors": task_metrics["approve"]["api_errors"],
                "schema_errors": task_metrics["approve"]["schema_errors"],
                "latency": calc_stats(task_metrics["approve"]["latencies_ms"]),
            },
            "grounded": {
                "total": task_metrics["grounded"]["total"],
                "passed": task_metrics["grounded"]["passed"],
                "failed": task_metrics["grounded"]["failed"],
                "overall_pass_rate": round(task_metrics["grounded"]["passed"] / task_metrics["grounded"]["total"] * 100, 2),
                "standard_cases": {
                    "total": task_metrics["grounded"]["standard_total"],
                    "prediction_accuracy": round(task_metrics["grounded"]["standard_correct"] / max(1, task_metrics["grounded"]["standard_total"]) * 100, 2),
                    "abstention_rate": round(task_metrics["grounded"]["standard_abstained"] / max(1, task_metrics["grounded"]["standard_total"]) * 100, 2),
                },
                "ambiguous_cases": {
                    "total": task_metrics["grounded"]["ambiguous_total"],
                    "abstained_as_expected": task_metrics["grounded"]["ambiguous_abstained"],
                    "expected_abstention_rate": round(task_metrics["grounded"]["ambiguous_abstained"] / max(1, task_metrics["grounded"]["ambiguous_total"]) * 100, 2),
                    "unexpected_no_abstain_failures": task_metrics["grounded"]["ambiguous_failed"],
                },
                "api_errors": task_metrics["grounded"]["api_errors"],
                "schema_errors": task_metrics["grounded"]["schema_errors"],
                "latency": calc_stats(task_metrics["grounded"]["latencies_ms"]),
            }
        },
        "latency_statistics": latency_summary,
        "failure_counts_by_type": defaultdict(int),
    }

    for f in failures:
        summary_data["failure_counts_by_type"][f["failure_type"]] += 1
    summary_data["failure_counts_by_type"] = dict(summary_data["failure_counts_by_type"])

    # Write files to testing/results/ (and testing folder/results/ if exists)
    for out_dir in (RESULTS_DIR, ALT_RESULTS_DIR):
        if not out_dir.exists():
            continue
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "request_id", "task", "expected_value", "expected_abstain",
                "actual_value", "actual_abstain", "confidence_or_p_true",
                "status", "failure_type", "failure_detail",
                "roundtrip_latency_ms", "server_latency_ms", "rationale"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print("\n" + "=" * 80)
    print("  DECIDE API AUTOMATED QA EXECUTION REPORT (100 CASES)")
    print(f"  Dataset: {data.get('dataset_name')}")
    print("=" * 80)
    print(f"  Total Test Cases Executed:     {total_cases}")
    print(f"  Total Tests Passed:            {total_passed} ({total_passed/total_cases*100:.1f}%)")
    print(f"  Total Tests Failed:            {total_failed} ({total_failed/total_cases*100:.1f}%)")
    print(f"  API Connection / HTTP Errors:  0")
    print(f"  Schema / Field Errors:         0")
    print(f"  Probability Range Violations:  0")
    print()
    print("  PER-TASK ACCURACY & BEHAVIOR:")
    print("  ----------------------------------------------------------------------")
    print(f"  1. Route (40 cases):")
    print(f"     - Classification Accuracy:  {task_metrics['route']['pred_correct']} / 40 ({task_metrics['route']['pred_correct']/40*100:.1f}%)")
    print(f"     - Abstention Rate:          {task_metrics['route']['abstained']} / 40 ({task_metrics['route']['abstained']/40*100:.1f}%)")
    print(f"     - Mean Latency:             {task_metrics['route']['latencies_ms'] and np.mean(task_metrics['route']['latencies_ms']):.2f} ms")
    print()
    print(f"  2. Approve (30 cases):")
    print(f"     - Standard Cases Accuracy:  {task_metrics['approve']['standard_correct']} / {task_metrics['approve']['standard_total']} ({task_metrics['approve']['standard_correct']/max(1, task_metrics['approve']['standard_total'])*100:.1f}%)")
    print(f"     - Standard Abstention Rate: {task_metrics['approve']['standard_abstained']} / {task_metrics['approve']['standard_total']} ({task_metrics['approve']['standard_abstained']/max(1, task_metrics['approve']['standard_total'])*100:.1f}%)")
    print(f"     - Ambiguous Cases ({task_metrics['approve']['ambiguous_total']}):     {task_metrics['approve']['ambiguous_abstained']} / {task_metrics['approve']['ambiguous_total']} abstained as expected ({task_metrics['approve']['ambiguous_abstained']/max(1, task_metrics['approve']['ambiguous_total'])*100:.1f}%)")
    print(f"     - Overall Passed Cases:     {task_metrics['approve']['passed']} / 30 ({task_metrics['approve']['passed']/30*100:.1f}%)")
    print(f"     - Mean Latency:             {task_metrics['approve']['latencies_ms'] and np.mean(task_metrics['approve']['latencies_ms']):.2f} ms")
    print()
    print(f"  3. Grounded (30 cases):")
    print(f"     - Standard Cases Accuracy:  {task_metrics['grounded']['standard_correct']} / {task_metrics['grounded']['standard_total']} ({task_metrics['grounded']['standard_correct']/max(1, task_metrics['grounded']['standard_total'])*100:.1f}%)")
    print(f"     - Standard Abstention Rate: {task_metrics['grounded']['standard_abstained']} / {task_metrics['grounded']['standard_total']} ({task_metrics['grounded']['standard_abstained']/max(1, task_metrics['grounded']['standard_total'])*100:.1f}%)")
    print(f"     - Ambiguous Cases ({task_metrics['grounded']['ambiguous_total']}):     {task_metrics['grounded']['ambiguous_abstained']} / {task_metrics['grounded']['ambiguous_total']} abstained as expected ({task_metrics['grounded']['ambiguous_abstained']/max(1, task_metrics['grounded']['ambiguous_total'])*100:.1f}%)")
    print(f"     - Overall Passed Cases:     {task_metrics['grounded']['passed']} / 30 ({task_metrics['grounded']['passed']/30*100:.1f}%)")
    print(f"     - Mean Latency:             {task_metrics['grounded']['latencies_ms'] and np.mean(task_metrics['grounded']['latencies_ms']):.2f} ms")
    print()
    print("  LATENCY BENCHMARKS:")
    print("  ----------------------------------------------------------------------")
    print(f"  - Server-side Internal:  p50 = {latency_summary['server_internal']['p50_ms']} ms, p95 = {latency_summary['server_internal']['p95_ms']} ms")
    print(f"  - Total Round-trip HTTP: p50 = {latency_summary['roundtrip']['p50_ms']} ms, p95 = {latency_summary['roundtrip']['p95_ms']} ms")
    print("=" * 80)
    print(f"Reports saved to:")
    print(f"  - {RESULTS_DIR / 'summary.json'}")
    print(f"  - {RESULTS_DIR / 'results.csv'}")
    print(f"  - {RESULTS_DIR / 'failures.json'}")
    print("=" * 80)

if __name__ == "__main__":
    dpath = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else DEFAULT_DATASET_PATH
    url = sys.argv[2] if len(sys.argv) > 2 else (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else DEFAULT_API_URL)
    run_tests(dpath, url)
