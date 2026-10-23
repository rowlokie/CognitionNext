"""
Experiment: train grounded head with contextual numeric entity features.
Test on dev set and on the 13 user cases.
"""
import sys, json, pickle
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from proto_numeric import compare_entities, extract_entities
from baseline import (
    load_jsonl_safe, clean_text, safe_str, tier_onehot, amount_features,
    serialize_grounded, enriched_grounding_features
)

ROOT = Path(".")

# Load data
train_records = load_jsonl_safe(str(ROOT / "data" / "train.jsonl"))
dev_records   = load_jsonl_safe(str(ROOT / "data" / "dev.jsonl"))

np.random.seed(42)
n_total = len(train_records)
n_calib = max(int(n_total * 0.1), 100)
indices  = np.random.permutation(n_total)
calib_idx = set(indices[:n_calib].tolist())
fit_records = [train_records[i] for i in range(n_total) if i not in calib_idx]

with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
    prod_model = pickle.load(f)

shared_word = prod_model.shared_tfidf_word
shared_char = prod_model.shared_tfidf_char

def new_grounded_features(reply: str, article: str) -> np.ndarray:
    base_f = enriched_grounding_features(reply, article)
    num_comp = compare_entities(reply, article)
    numeric_f = np.array([
        num_comp["num_exact_matches"],
        num_comp["num_contradictions"],
        num_comp["unsupported_claims"],
        num_comp["max_contradiction_severity"],
        num_comp["has_numeric_contradiction"],
    ], dtype=np.float32)
    return np.concatenate([base_f, numeric_f])

def new_structured_grounded(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        new_grounded_features(reply, article),
    ])

print("Building feature matrices...")
fit_texts   = [serialize_grounded(r["state"]) for r in fit_records]
fit_structs = [new_structured_grounded(r["state"]) for r in fit_records]
y_fit       = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])

X_fit = hstack([
    shared_word.transform(fit_texts),
    shared_char.transform(fit_texts),
    csr_matrix(np.array(fit_structs, dtype=np.float32)),
])

print(f"X_fit shape: {X_fit.shape}")

print("Training CalibratedClassifierCV...")
base_clf = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", solver="saga")
cand_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
cand_clf.fit(X_fit, y_fit)

dev_texts   = [serialize_grounded(r["state"]) for r in dev_records]
dev_structs = [new_structured_grounded(r["state"]) for r in dev_records]
y_dev       = np.array([int(bool(r["labels"]["grounded"])) for r in dev_records])

X_dev = hstack([
    shared_word.transform(dev_texts),
    shared_char.transform(dev_texts),
    csr_matrix(np.array(dev_structs, dtype=np.float32)),
])

dev_probs = cand_clf.predict_proba(X_dev)[:, 1]

# Find best threshold on dev
best_th, best_f1 = 0.50, -1.0
for th in np.arange(0.35, 0.70, 0.01):
    preds = (dev_probs >= th).astype(int)
    f = f1_score(y_dev, preds, average="macro")
    if f > best_f1:
        best_f1, best_th = f, th

print(f"Dev Macro-F1: {best_f1*100:.2f}% at threshold={best_th:.3f}")

# Test the 13 user cases
TESTS = [
    ("Test 1", False, "Standard and Pro warranties exclude liquid ingress and submersion.", "Pro warranty covers water damage caused by full immersion in a swimming pool."),
    ("Test 2", False, "Care+ is optional; applicable charges are specified in the customer's policy.", "Care+ has a zero deductible."),
    ("Test 7", False, "Returns allowed within 14 calendar days, subject to conditions and exclusions.", "Returns allowed within 30 days with no exceptions."),
    ("Test 8", False, "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.", "Refund processed within 30 calendar days."),
    ("Test 9", True, "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.", "Approved refunds processed within 7 calendar days after approval; bank posting times may vary."),
    ("Test 10", False, "Standard warranty lasts 12 months.", "Standard warranty lasts 24 months."),
    ("Test 11", True, "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions.", "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions."),
    ("Test 12", False, "Free replacements are limited to confirmed manufacturing defects; accidental and physical damage are excluded.", "Guarantees a free replacement for every damaged device regardless of cause."),
    ("Test 13", False, "Replacement fee is $100.", "Replacement fee is $250."),
]

print("\nEvaluating user cases with candidate model:")
print(f"{'ID':<8} {'Expected':<10} {'Predicted':<10} {'p_true':<10} {'Status'}")
print("-" * 55)
for tid, exp, kb, reply in TESTS:
    state = {"customer_tier": "pro", "amount": 0.0, "draft_reply": reply, "kb_article": kb}
    t = serialize_grounded(state)
    s = new_structured_grounded(state)
    x = hstack([shared_word.transform([t]), shared_char.transform([t]), csr_matrix([s])])
    p = cand_clf.predict_proba(x)[0, 1]
    pred = p >= best_th
    status = "PASS" if pred == exp else "FAIL"
    print(f"{tid:<8} {str(exp):<10} {str(pred):<10} {p:<10.4f} {status}")
