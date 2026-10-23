"""
Test feature representations and model weights with contradiction interaction terms.
"""
import sys, json, pickle
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

from proto_numeric_refined import compare_entities_refined
from baseline import (
    load_jsonl_safe, clean_text, safe_str, tier_onehot, amount_features,
    serialize_grounded, enriched_grounding_features
)

ROOT = Path(".")
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

def comprehensive_grounded_features(reply: str, article: str) -> np.ndarray:
    base_f = enriched_grounding_features(reply, article)
    # base_f indices:
    # 0: jaccard, 1: coverage_r, 2: coverage_a, 3: len_ratio_log
    # 4: prec_content, 5: rec_content
    # 6: polarity_contradiction
    jaccard = base_f[0]
    coverage_r = base_f[1]
    polarity_contra = base_f[6]

    num_comp = compare_entities_refined(reply, article)
    has_num_contra = num_comp["has_numeric_contradiction"]
    num_contra = num_comp["num_contradictions"]
    num_matches = num_comp["num_exact_matches"]
    num_sev = num_comp["max_contradiction_severity"]
    unsupported = num_comp["unsupported_claims"]

    # Explicit contradiction interaction terms:
    # High lexical overlap + contradiction = deceptive ungrounded
    num_contra_overlap = has_num_contra * coverage_r
    num_contra_severity_overlap = num_sev * coverage_r
    polarity_contra_overlap = polarity_contra * coverage_r

    # High numeric match is strong grounded signal
    num_match_signal = float(num_matches > 0 and has_num_contra == 0)

    # Combined contradiction flag
    any_contra = float(has_num_contra > 0 or polarity_contra > 0)
    contra_intensity = (has_num_contra * 2.0 + num_sev + polarity_contra * 2.0)

    # Net alignment score: (supported overlap) - (contradiction penalty)
    net_alignment = (coverage_r + jaccard) * (1.0 - min(contra_intensity, 1.0))

    numeric_block = np.array([
        num_matches,
        num_contra,
        unsupported,
        num_sev,
        has_num_contra,
        num_contra_overlap,
        num_contra_severity_overlap,
        polarity_contra_overlap,
        num_match_signal,
        any_contra,
        contra_intensity,
        net_alignment,
    ], dtype=np.float32)

    return np.concatenate([base_f, numeric_block])

def build_structured(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        comprehensive_grounded_features(reply, article),
    ])

print("Building feature matrices...")
fit_texts   = [serialize_grounded(r["state"]) for r in fit_records]
fit_structs = [build_structured(r["state"]) for r in fit_records]
y_fit       = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])

X_fit = hstack([
    shared_word.transform(fit_texts),
    shared_char.transform(fit_texts),
    csr_matrix(np.array(fit_structs, dtype=np.float32)),
])

print("Training CalibratedClassifierCV...")
base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", solver="saga")
cand_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
cand_clf.fit(X_fit, y_fit)

dev_texts   = [serialize_grounded(r["state"]) for r in dev_records]
dev_structs = [build_structured(r["state"]) for r in dev_records]
y_dev       = np.array([int(bool(r["labels"]["grounded"])) for r in dev_records])

X_dev = hstack([
    shared_word.transform(dev_texts),
    shared_char.transform(dev_texts),
    csr_matrix(np.array(dev_structs, dtype=np.float32)),
])

dev_probs = cand_clf.predict_proba(X_dev)[:, 1]

best_th, best_f1 = 0.50, -1.0
for th in np.arange(0.35, 0.70, 0.01):
    preds = (dev_probs >= th).astype(int)
    f = f1_score(y_dev, preds, average="macro")
    if f > best_f1:
        best_f1, best_th = f, th

print(f"Dev Macro-F1: {best_f1*100:.2f}% at threshold={best_th:.3f}")

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
    s = build_structured(state)
    x = hstack([shared_word.transform([t]), shared_char.transform([t]), csr_matrix([s])])
    p = cand_clf.predict_proba(x)[0, 1]
    pred = p >= best_th
    status = "PASS" if pred == exp else "FAIL"
    print(f"{tid:<8} {str(exp):<10} {str(pred):<10} {p:<10.4f} {status}")
