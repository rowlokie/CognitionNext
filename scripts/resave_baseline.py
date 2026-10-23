#!/usr/bin/env python3
"""Helper to re-save baseline model so it loads from 'baseline' module correctly."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Import baseline as module (not __main__)
import baseline
from baseline import BaselineModel, load_jsonl_safe, ROUTE_CLASSES
import numpy as np
import pickle

ARTIFACTS_DIR = ROOT / "artifacts"

print("Re-saving baseline model with correct module path...")

train_path = ROOT / "data" / "train.jsonl"
dev_path = ROOT / "data" / "dev.jsonl"
train_records = load_jsonl_safe(str(train_path))
dev_records = load_jsonl_safe(str(dev_path))

np.random.seed(42)
n_total = len(train_records)
n_calib = max(int(n_total * 0.1), 100)
indices = np.random.permutation(n_total)
calib_idx = set(indices[:n_calib])
fit_records = [train_records[i] for i in range(n_total) if i not in calib_idx]
calib_records = [train_records[i] for i in range(n_total) if i in calib_idx]

print(f"  Fit: {len(fit_records)}, Calib: {len(calib_records)}")

model = BaselineModel()
model.fit(fit_records)
model.save(ARTIFACTS_DIR)
print("  Saved baseline_model.pkl")

# Quick verify
with open(ARTIFACTS_DIR / "baseline_model.pkl", "rb") as f:
    m2 = pickle.load(f)
print("  Verified load OK")

# Quick dev metrics
print("\nQuick dev eval:")
correct_route = 0
route_preds, route_golds = [], []
approve_tp, approve_fp, approve_fn = 0, 0, 0
grounded_tp, grounded_fp, grounded_fn = 0, 0, 0

for r in dev_records:
    state = r.get("state", {}) or {}
    labels = r.get("labels", {}) or {}

    res = m2.predict(state, "route", r.get("request_id", ""))
    route_preds.append(res["value"])
    route_golds.append(labels.get("route"))

    res = m2.predict(state, "approve", r.get("request_id", ""))
    pred = res["value"]
    gold = bool(labels.get("approve", False))
    if pred and gold: approve_tp += 1
    elif pred and not gold: approve_fp += 1
    elif not pred and gold: approve_fn += 1

    res = m2.predict(state, "grounded", r.get("request_id", ""))
    pred = res["value"]
    gold = bool(labels.get("grounded", False))
    if pred and gold: grounded_tp += 1
    elif pred and not gold: grounded_fp += 1
    elif not pred and gold: grounded_fn += 1

f1s = []
for cls in ROUTE_CLASSES:
    tp = sum(1 for p, g in zip(route_preds, route_golds) if p == cls and g == cls)
    fp = sum(1 for p, g in zip(route_preds, route_golds) if p == cls and g != cls)
    fn = sum(1 for p, g in zip(route_preds, route_golds) if p != cls and g == cls)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1s.append(f1)
macro_f1_route = sum(f1s) / len(f1s)

prec_a = approve_tp / (approve_tp + approve_fp) if (approve_tp + approve_fp) > 0 else 0
rec_a = approve_tp / (approve_tp + approve_fn) if (approve_tp + approve_fn) > 0 else 0
f1_approve = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0
cost = 3 * approve_fp + approve_fn

prec_g = grounded_tp / (grounded_tp + grounded_fp) if (grounded_tp + grounded_fp) > 0 else 0
rec_g = grounded_tp / (grounded_tp + grounded_fn) if (grounded_tp + grounded_fn) > 0 else 0
f1_grounded = 2 * prec_g * rec_g / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0

print(f"  route macro-F1:    {macro_f1_route:.4f}")
print(f"  approve F1:        {f1_approve:.4f}")
print(f"  approve cost:      {cost}")
print(f"  grounded macro-F1: {f1_grounded:.4f}")
print("\nDone. Now run: python scripts/phase3_calibration.py")
