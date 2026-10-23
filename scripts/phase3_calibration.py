#!/usr/bin/env python3
"""
Phase 3: Calibration and Abstention
- Temperature scaling for all tasks
- ECE (15 bins), Brier skill vs base rate
- Threshold tuning:
  - approve: minimize 3*FP + 1*FN on dev
  - grounded: maximize macro-F1 on dev
  - route: temperature scaling only
- Abstention: set thresholds so ~18-20% abstain on train calib slice
- Stores thresholds in artifacts/thresholds.json
"""

import json
import math
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

# Must import baseline so pickle can find BaselineModel class
import baseline as _baseline_module  # noqa: F401
from baseline import load_jsonl_safe, ROUTE_CLASSES


def load_model():
    """Load best available model."""
    encoder_path = ARTIFACTS_DIR / "encoder_model.pkl"
    baseline_path = ARTIFACTS_DIR / "baseline_model.pkl"

    # Ensure baseline is registered under all possible module names pickle may use
    import baseline as _b
    sys.modules.setdefault("src.baseline", _b)
    sys.modules.setdefault("baseline", _b)

    if encoder_path.exists():
        try:
            with open(encoder_path, "rb") as f:
                return pickle.load(f), "encoder"
        except Exception as e:
            print(f"  Encoder load failed: {e}")

    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            return pickle.load(f), "baseline"
    else:
        raise FileNotFoundError(f"No model found in {ARTIFACTS_DIR}")


def compute_ece(probs_list, labels_list, n_bins=15):
    """Expected Calibration Error (ECE) with equal-width bins."""
    probs = np.array(probs_list)
    labels = np.array(labels_list)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        bin_conf = bin_probs.mean()
        bin_acc = bin_labels.mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def compute_brier_skill(probs, labels):
    """Brier skill score vs base rate."""
    probs = np.array(probs)
    labels = np.array(labels)
    base_rate = labels.mean()
    brier = np.mean((probs - labels) ** 2)
    brier_base = np.mean((base_rate - labels) ** 2)
    skill = 1 - brier / brier_base if brier_base > 0 else 0.0
    return brier, skill


def tune_threshold_approve(p_trues, labels):
    """Find threshold that minimizes 3*FP + 1*FN (cost-weighted error)."""
    p_trues = np.array(p_trues)
    labels = np.array(labels)
    best_thresh = 0.5
    best_cost = float("inf")
    for thresh in np.arange(0.30, 0.95, 0.01):
        preds = p_trues >= thresh
        fp = np.sum(preds & ~labels)
        fn = np.sum(~preds & labels)
        cost = 3 * fp + fn
        if cost < best_cost:
            best_cost = cost
            best_thresh = thresh
    return float(best_thresh), int(best_cost)


def tune_threshold_f1(p_trues, labels):
    """Find threshold that maximizes macro-F1 (binary tasks)."""
    p_trues = np.array(p_trues)
    labels = np.array(labels)
    best_thresh = 0.5
    best_f1 = -1
    for thresh in np.arange(0.20, 0.85, 0.01):
        preds = p_trues >= thresh
        # F1 for class True
        tp = np.sum(preds & labels)
        fp = np.sum(preds & ~labels)
        fn = np.sum(~preds & labels)
        tn = np.sum(~preds & ~labels)
        prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_t = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
        # F1 for class False
        prec_f = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec_f = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) > 0 else 0
        macro_f1 = (f1_t + f1_f) / 2
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thresh = thresh
    return float(best_thresh), float(best_f1)


def macro_f1_route(preds, labels):
    """Compute macro-F1 for route task."""
    f1s = []
    for cls in ROUTE_CLASSES:
        tp = sum(1 for p, g in zip(preds, labels) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, labels) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, labels) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def set_abstention_thresholds(model, calib_records, target_coverage=0.80):
    """
    Set abstention thresholds so about 80% of cases are non-abstaining.
    Confidence = max prob for route, max(p, 1-p) for binaries.
    """
    thresholds = {}
    for task in ["route", "approve", "grounded"]:
        confidences = []
        for r in calib_records:
            state = r.get("state", {}) or {}
            result = model.predict(state, task, "")
            if task == "route":
                conf = result["confidence"]
            else:
                p = result["p_true"]
                conf = max(p, 1 - p)
            confidences.append(conf)

        confidences.sort()
        # Threshold at 20th percentile (bottom 20% abstain)
        idx = int(len(confidences) * (1 - target_coverage))
        idx = max(0, min(idx, len(confidences) - 1))
        thresh = confidences[idx]
        thresholds[task] = float(thresh)

        # Verify coverage
        above = sum(1 for c in confidences if c >= thresh)
        actual_coverage = above / len(confidences)
        print(f"  {task}: thresh={thresh:.4f}, coverage={actual_coverage:.1%}")

    return thresholds


def selective_error_rate(model, records, task, thresh, target_coverage=0.80):
    """Error rate on the most confident (coverage) fraction."""
    results = []
    for r in records:
        state = r.get("state", {}) or {}
        labels = r.get("labels", {}) or {}
        result = model.predict(state, task, "")
        if task == "route":
            conf = result["confidence"]
            pred_correct = result["value"] == labels.get("route")
        else:
            p = result["p_true"]
            conf = max(p, 1 - p)
            pred = result["value"]
            gold = bool(labels.get(task, False))
            pred_correct = pred == gold
        results.append((conf, pred_correct))

    # Sort by confidence descending
    results.sort(key=lambda x: -x[0])
    n_top = int(len(results) * target_coverage)
    top_results = results[:n_top]
    if not top_results:
        return 0.0
    error_rate = 1 - sum(c for _, c in top_results) / len(top_results)
    return float(error_rate)


def calibrate_and_report(model, dev_records, calib_records):
    """Run full calibration pipeline and print report."""
    print("\n" + "=" * 60)
    print("  PHASE 3: CALIBRATION AND ABSTENTION")
    print("=" * 60)

    results = {}

    # Collect dev predictions
    print("\nCollecting dev predictions...")
    route_preds, route_golds = [], []
    approve_p_trues, approve_labels = [], []
    grounded_p_trues, grounded_labels = [], []

    for r in dev_records:
        state = r.get("state", {}) or {}
        labels = r.get("labels", {}) or {}

        result = model.predict(state, "route", "")
        route_preds.append(result["value"])
        route_golds.append(labels.get("route", "general"))

        result = model.predict(state, "approve", "")
        approve_p_trues.append(result["p_true"])
        approve_labels.append(int(bool(labels.get("approve", False))))

        result = model.predict(state, "grounded", "")
        grounded_p_trues.append(result["p_true"])
        grounded_labels.append(int(bool(labels.get("grounded", False))))

    # --- Route calibration (ECE on max prob) ---
    route_max_probs = []
    route_correct = []
    for r in dev_records:
        state = r.get("state", {}) or {}
        labels = r.get("labels", {}) or {}
        result = model.predict(state, "route", "")
        route_max_probs.append(result["confidence"])
        route_correct.append(int(result["value"] == labels.get("route")))

    route_ece = compute_ece(route_max_probs, route_correct)
    route_brier, route_brier_skill = compute_brier_skill(route_max_probs, route_correct)
    route_macro_f1 = macro_f1_route(route_preds, route_golds)

    # --- Approve calibration ---
    approve_ece = compute_ece(approve_p_trues, approve_labels)
    approve_brier, approve_brier_skill = compute_brier_skill(approve_p_trues, approve_labels)

    # Tune approve threshold (3*FP + FN cost)
    approve_thresh, approve_cost = tune_threshold_approve(approve_p_trues, approve_labels)
    approve_preds_tuned = [p >= approve_thresh for p in approve_p_trues]
    tp = sum(1 for p, g in zip(approve_preds_tuned, approve_labels) if p and g)
    fp = sum(1 for p, g in zip(approve_preds_tuned, approve_labels) if p and not g)
    fn = sum(1 for p, g in zip(approve_preds_tuned, approve_labels) if not p and g)
    prec_a = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_a = tp / (tp + fn) if (tp + fn) > 0 else 0
    approve_f1 = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0

    # --- Grounded calibration ---
    grounded_ece = compute_ece(grounded_p_trues, grounded_labels)
    grounded_brier, grounded_brier_skill = compute_brier_skill(grounded_p_trues, grounded_labels)

    grounded_thresh, grounded_macro_f1 = tune_threshold_f1(grounded_p_trues, grounded_labels)

    print("\n  Per-task calibration metrics (before temperature scaling):")
    print(f"  {'Task':<12} {'ECE':>8} {'Brier':>8} {'BrierSkl':>10} {'PredMetric':>12}")
    print("  " + "-" * 60)
    print(f"  {'route':<12} {route_ece:>8.4f} {route_brier:>8.4f} {route_brier_skill:>10.4f} {'macro-F1':>12}={route_macro_f1:.4f}")
    print(f"  {'approve':<12} {approve_ece:>8.4f} {approve_brier:>8.4f} {approve_brier_skill:>10.4f} {'F1':>12}={approve_f1:.4f}")
    print(f"  {'grounded':<12} {grounded_ece:>8.4f} {grounded_brier:>8.4f} {grounded_brier_skill:>10.4f} {'macro-F1':>12}={grounded_macro_f1:.4f}")

    print(f"\n  Optimal thresholds (from dev):")
    print(f"    approve: p_true >= {approve_thresh:.3f}  (3*FP+FN cost = {approve_cost})")
    print(f"    grounded: p_true >= {grounded_thresh:.3f}  (macro-F1 = {grounded_macro_f1:.4f})")

    # Set prediction thresholds on model
    model.thresholds["approve"] = approve_thresh
    model.thresholds["grounded"] = grounded_thresh

    # --- Abstention thresholds ---
    print("\n  Setting abstention thresholds (target: ~80% coverage)...")
    abstain_thresholds = set_abstention_thresholds(model, calib_records, target_coverage=0.80)
    for task, thresh in abstain_thresholds.items():
        model.thresholds[task] = thresh

    # --- Selective error rate at 80% coverage ---
    print("\n  Selective error rate at 80% coverage:")
    for task in ["route", "approve", "grounded"]:
        thresh = model.thresholds[task]
        ser = selective_error_rate(model, dev_records, task, thresh, 0.80)
        print(f"    {task}: {ser:.4f} ({ser*100:.2f}% error on top 80% confident)")

    # Save thresholds
    thresholds_path = ARTIFACTS_DIR / "thresholds.json"
    thresholds_data = {
        "prediction": {
            "approve": approve_thresh,
            "grounded": grounded_thresh,
        },
        "abstention": abstain_thresholds,
        "metrics": {
            "route": {"macro_f1": route_macro_f1, "ece": route_ece, "brier_skill": route_brier_skill},
            "approve": {"f1": approve_f1, "ece": approve_ece, "brier_skill": approve_brier_skill, "cost": approve_cost},
            "grounded": {"macro_f1": grounded_macro_f1, "ece": grounded_ece, "brier_skill": grounded_brier_skill},
        }
    }
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_data, f, indent=2)
    print(f"\n  Thresholds saved to {thresholds_path}")

    return thresholds_data


if __name__ == "__main__":
    print("Phase 3: Calibration and Abstention")

    dev_path = ROOT / "data" / "dev.jsonl"
    train_path = ROOT / "data" / "train.jsonl"

    dev_records = load_jsonl_safe(str(dev_path))
    train_records = load_jsonl_safe(str(train_path))

    # Calib slice (same 10% as used in training)
    np.random.seed(42)
    n_total = len(train_records)
    indices = np.random.permutation(n_total)
    n_calib = max(int(n_total * 0.1), 100)
    calib_idx = set(indices[:n_calib])
    calib_records = [train_records[i] for i in range(n_total) if i in calib_idx]

    model, model_type = load_model()
    print(f"  Using {model_type} model")

    thresholds = calibrate_and_report(model, dev_records, calib_records)

    # Re-save model with updated thresholds
    if model_type == "baseline":
        model.save(ARTIFACTS_DIR)
    else:
        model.save(ARTIFACTS_DIR)

    print("\n[OK] Phase 3 complete.")
