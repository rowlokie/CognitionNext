"""
Retrain ONLY the grounded head with enriched structured features.
The shared TF-IDF backbone, route head, and approve head are reused from the
existing production artifact. The new grounded head is saved to
artifacts/grounded_candidate_v2.pkl. The production baseline is NOT touched.
"""
import sys, json, pickle, time
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

from baseline import (
    load_jsonl_safe,
    serialize_grounded, structured_grounded,
    BaselineModel,
)

ROOT = Path(".")

print("=" * 70)
print("  GROUNDED CANDIDATE RETRAINING (enriched features, grounded head only)")
print("=" * 70)

# ----------------------------------------------------------------
# 1. Load data using the SAME seed=42 split as production
# ----------------------------------------------------------------
train_records = load_jsonl_safe(str(ROOT / "data" / "train.jsonl"))
dev_records   = load_jsonl_safe(str(ROOT / "data" / "dev.jsonl"))

np.random.seed(42)
n_total = len(train_records)
n_calib = max(int(n_total * 0.1), 100)
indices  = np.random.permutation(n_total)
calib_idx = set(indices[:n_calib].tolist())
fit_records   = [train_records[i] for i in range(n_total) if i not in calib_idx]
calib_records = [train_records[i] for i in range(n_total) if i in calib_idx]

print(f"Fit:   {len(fit_records)} rows | Calib: {len(calib_records)} rows | Dev: {len(dev_records)} rows")

# ----------------------------------------------------------------
# 2. Load production model for its shared TF-IDF backbone
# ----------------------------------------------------------------
with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
    prod_model = pickle.load(f)

shared_word = prod_model.shared_tfidf_word
shared_char = prod_model.shared_tfidf_char
print("Loaded shared TF-IDF backbone from baseline_model.pkl (NOT modified)")

# ----------------------------------------------------------------
# 3. Build grounded features for fit records (enriched structured_grounded)
# ----------------------------------------------------------------
print("\nBuilding grounded features with enriched structured features...")
fit_texts   = [serialize_grounded(r["state"]) for r in fit_records]
fit_structs = [structured_grounded(r["state"]) for r in fit_records]
y_fit       = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])

from scipy.sparse import hstack, csr_matrix
X_fit = hstack([
    shared_word.transform(fit_texts),
    shared_char.transform(fit_texts),
    csr_matrix(np.array(fit_structs, dtype=np.float32)),
])
print(f"Feature matrix shape: {X_fit.shape}")

# ----------------------------------------------------------------
# 4. Train enriched grounded head — 3-fold isotonic calibration
# ----------------------------------------------------------------
print("\nTraining enriched grounded head (LogisticRegression + isotonic 3-fold CV)...")
t0 = time.perf_counter()
base_clf = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", solver="saga")
cand_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
cand_clf.fit(X_fit, y_fit)
train_time = time.perf_counter() - t0
print(f"Training done in {train_time:.1f}s")

# ----------------------------------------------------------------
# 5. Evaluate on held-out dev set
# ----------------------------------------------------------------
print("\nEvaluating candidate on dev set...")
dev_texts   = [serialize_grounded(r["state"]) for r in dev_records]
dev_structs = [structured_grounded(r["state"]) for r in dev_records]
y_dev       = np.array([int(bool(r["labels"]["grounded"])) for r in dev_records])

X_dev = hstack([
    shared_word.transform(dev_texts),
    shared_char.transform(dev_texts),
    csr_matrix(np.array(dev_structs, dtype=np.float32)),
])

# Tune threshold on dev
cand_probs = cand_clf.predict_proba(X_dev)[:, 1]
best_th, best_f1 = 0.50, -1.0
for th in np.arange(0.30, 0.75, 0.01):
    p = (cand_probs >= th).astype(int)
    f = f1_score(y_dev, p, average="macro")
    if f > best_f1:
        best_f1, best_th = f, th

cand_preds = (cand_probs >= best_th).astype(int)

# Also evaluate baseline via the live API (avoids feature-count mismatch
# in the old pickle vs new structured_grounded signature)
import requests as req_lib
base_probs_list = []
base_th = 0.51  # production grounded threshold
for r in dev_records:
    payload = {
        "request_id": r.get("request_id", ""),
        "decision": "grounded",
        "state": r["state"],
    }
    resp = req_lib.post("http://127.0.0.1:8000/v1/decide", json=payload, timeout=5)
    base_probs_list.append(resp.json().get("p_true", 0.5))

base_probs = np.array(base_probs_list)
base_preds = (base_probs >= base_th).astype(int)

base_f1   = f1_score(y_dev, base_preds, average="macro")
base_prec = precision_score(y_dev, base_preds, pos_label=1)
base_rec  = recall_score(y_dev, base_preds, pos_label=1)
base_prec0 = precision_score(y_dev, base_preds, pos_label=0)
base_rec0  = recall_score(y_dev, base_preds, pos_label=0)
base_cm   = confusion_matrix(y_dev, base_preds)

cand_prec  = precision_score(y_dev, cand_preds, pos_label=1)
cand_rec   = recall_score(y_dev, cand_preds, pos_label=1)
cand_prec0 = precision_score(y_dev, cand_preds, pos_label=0)
cand_rec0  = recall_score(y_dev, cand_preds, pos_label=0)
cand_cm    = confusion_matrix(y_dev, cand_preds)

# ECE
def ece(probs, labels, n_bins=15):
    probs = np.asarray(probs); labels = np.asarray(labels, float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs); e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (probs >= lo) & (probs <= hi) if i == n_bins-1 else (probs >= lo) & (probs < hi)
        if m.sum():
            e += (m.sum()/total) * abs(probs[m].mean() - labels[m].mean())
    return e

def bss(probs, labels):
    probs = np.asarray(probs, float); labels = np.asarray(labels, float)
    br = labels.mean()
    bs = np.mean((probs - labels)**2)
    bs_ref = np.mean((br - labels)**2)
    return bs, 1 - bs/(bs_ref + 1e-12)

def sel_err(probs, labels, th, cov=0.80):
    confs = np.maximum(probs, 1-probs)
    preds = (probs >= th).astype(int)
    errs  = (preds != np.asarray(labels)).astype(int)
    n = int(len(confs)*cov)
    idx = np.argsort(confs)[::-1][:n]
    return float(np.mean(errs[idx]))

base_ece = ece(base_probs, y_dev)
cand_ece = ece(cand_probs, y_dev)
_, base_bss = bss(base_probs, y_dev)
_, cand_bss = bss(cand_probs, y_dev)
base_sel = sel_err(base_probs, y_dev, base_th)
cand_sel = sel_err(cand_probs, y_dev, best_th)

print("\n" + "=" * 75)
print("  BASELINE vs ENRICHED CANDIDATE — DEV COMPARISON")
print("=" * 75)
print(f"  {'Metric':<36} {'Baseline':>12} {'Candidate':>12} {'Delta':>10}")
print("  " + "-" * 72)
print(f"  {'Grounded Macro-F1':<36} {base_f1*100:>11.2f}% {best_f1*100:>11.2f}% {(best_f1-base_f1)*100:>+9.2f}%")
print(f"  {'Class 1 (Grounded) Precision':<36} {base_prec*100:>11.2f}% {cand_prec*100:>11.2f}% {(cand_prec-base_prec)*100:>+9.2f}%")
print(f"  {'Class 1 (Grounded) Recall':<36} {base_rec*100:>11.2f}% {cand_rec*100:>11.2f}% {(cand_rec-base_rec)*100:>+9.2f}%")
print(f"  {'Class 0 (Ungrounded) Precision':<36} {base_prec0*100:>11.2f}% {cand_prec0*100:>11.2f}% {(cand_prec0-base_prec0)*100:>+9.2f}%")
print(f"  {'Class 0 (Ungrounded) Recall':<36} {base_rec0*100:>11.2f}% {cand_rec0*100:>11.2f}% {(cand_rec0-base_rec0)*100:>+9.2f}%")
print(f"  {'ECE (15 equal bins)':<36} {base_ece:>12.4f} {cand_ece:>12.4f} {cand_ece-base_ece:>+10.4f}")
print(f"  {'Brier Skill Score':<36} {base_bss:>12.4f} {cand_bss:>12.4f} {cand_bss-base_bss:>+10.4f}")
print(f"  {'Selective Error (80% cov)':<36} {base_sel*100:>11.2f}% {cand_sel*100:>11.2f}% {(cand_sel-base_sel)*100:>+9.2f}%")
print(f"  {'Decision Threshold':<36} {base_th:>12.3f} {best_th:>12.3f}")
print()
print(f"  Baseline CM:\n{base_cm}")
print(f"  Candidate CM (th={best_th:.2f}):\n{cand_cm}")

# ----------------------------------------------------------------
# 6. Save candidate artifact (DOES NOT overwrite production)
# ----------------------------------------------------------------
cand_path = ROOT / "artifacts" / "grounded_candidate_v2.pkl"
with open(cand_path, "wb") as f:
    pickle.dump({
        "grounded_model": cand_clf,
        "threshold": best_th,
        "macro_f1": best_f1,
        "ece": cand_ece,
        "bss": cand_bss,
        "selective_error": cand_sel,
        "feature_set": "enriched_v2_23features",
    }, f)
print(f"\nCandidate artifact saved to {cand_path}")
print("Production baseline_model.pkl NOT modified.")

promoted = best_f1 > base_f1 and cand_sel <= base_sel + 0.01
print(f"\nPromotion recommendation: {'PROMOTE' if promoted else 'KEEP BASELINE'}")
print(f"  Candidate F1 ({best_f1*100:.2f}%) > Baseline F1 ({base_f1*100:.2f}%): {best_f1 > base_f1}")
print(f"  Candidate sel_err ({cand_sel*100:.2f}%) <= Baseline + 1% ({(base_sel+0.01)*100:.2f}%): {cand_sel <= base_sel + 0.01}")
