"""
Fast standalone test: gating coverage_a and testing all 9 user cases.
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

def fully_gated_features(reply: str, article: str) -> np.ndarray:
    base_f = enriched_grounding_features(reply, article)
    polarity_contra = base_f[6]

    num_comp = compare_entities_refined(reply, article)
    has_num_contra = num_comp["has_numeric_contradiction"]
    num_contra     = num_comp["num_contradictions"]
    num_matches    = num_comp["num_exact_matches"]
    num_sev        = num_comp["max_contradiction_severity"]
    unsupported    = num_comp["unsupported_claims"]

    contra_flag = float(has_num_contra > 0 or polarity_contra > 0.3)
    contra_intensity = min(has_num_contra * 2.0 + num_sev + polarity_contra * 2.0, 3.0)

    # When contradiction exists, ALL lexical overlap is suppressed
    gate = max(0.0, 1.0 - contra_intensity)
    gated_jaccard    = base_f[0] * gate
    gated_coverage_r = base_f[1] * gate
    gated_coverage_a = base_f[2] * gate
    gated_prec_c     = base_f[4] * gate
    gated_rec_c      = base_f[5] * gate
    gated_sent_min   = base_f[12] * gate
    gated_sent_mean  = base_f[13] * gate
    gated_sent_cov1  = base_f[19] * gate
    gated_sent_cov2  = base_f[20] * gate

    # Direct contradiction penalty features
    contra_penalty_1 = contra_flag * 30.0
    contra_penalty_2 = contra_intensity * 30.0
    numeric_mismatch_penalty = has_num_contra * 60.0
    short_reply_contra = float(len(reply.split()) <= 12 and has_num_contra > 0) * 40.0

    match_reward = float(num_matches > 0 and contra_flag == 0) * 10.0

    features = np.array([
        gated_jaccard,
        gated_coverage_r,
        gated_coverage_a,
        base_f[3],  # len_ratio
        gated_prec_c,
        gated_rec_c,
        polarity_contra,
        num_matches,
        num_contra,
        unsupported,
        num_sev,
        has_num_contra,
        contra_flag,
        contra_intensity,
        contra_penalty_1,
        contra_penalty_2,
        numeric_mismatch_penalty,
        short_reply_contra,
        match_reward,
        base_f[10], # neg_count_diff
        base_f[11], # article_neg_reply_positive
        gated_sent_min,
        gated_sent_mean,
        base_f[14], # unsupported_ratio
        base_f[16], # optional_entitlement_conflict
        gated_sent_cov1,
        gated_sent_cov2,
    ], dtype=np.float32)

    return features

def build_structured_full(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        fully_gated_features(reply, article),
    ])

print("Building feature matrices...")
fit_texts   = [serialize_grounded(r["state"]) for r in fit_records]
fit_structs = [build_structured_full(r["state"]) for r in fit_records]
y_fit       = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])

X_fit = hstack([
    shared_word.transform(fit_texts),
    shared_char.transform(fit_texts),
    csr_matrix(np.array(fit_structs, dtype=np.float32)),
])

print("Training CalibratedClassifierCV...")
base_clf = LogisticRegression(C=2.0, max_iter=500, class_weight="balanced", solver="lbfgs")
cand_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
cand_clf.fit(X_fit, y_fit)

dev_texts   = [serialize_grounded(r["state"]) for r in dev_records]
dev_structs = [build_structured_full(r["state"]) for r in dev_records]
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

print(f"\nDev Macro-F1: {best_f1*100:.2f}% at threshold={best_th:.3f}")

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

print("\n" + "=" * 65)
print("Evaluating user cases with fully gated candidate:")
print("=" * 65)
print(f"{'ID':<8} {'Expected':<10} {'Predicted':<10} {'p_true':<10} {'Status'}")
print("-" * 65)
all_pass = True
for tid, exp, kb, reply in TESTS:
    state = {"customer_tier": "pro", "amount": 0.0, "draft_reply": reply, "kb_article": kb}
    t = serialize_grounded(state)
    s = build_structured_full(state)
    x = hstack([shared_word.transform([t]), shared_char.transform([t]), csr_matrix([s])])
    p = cand_clf.predict_proba(x)[0, 1]
    pred = p >= best_th
    status = "PASS" if pred == exp else "FAIL"
    if status == "FAIL": all_pass = False
    print(f"{tid:<8} {str(exp):<10} {str(pred):<10} {p:<10.4f} {status}")

print("=" * 65)
print(f"ALL 9 USER TEST CASES PASS: {all_pass}")

# Save if all pass
if all_pass and best_f1 > 0.75:
    with open(ROOT / "artifacts" / "fully_gated_candidate.pkl", "wb") as f:
        pickle.dump({
            "grounded_model": cand_clf,
            "threshold": best_th,
            "macro_f1": best_f1,
            "build_structured": build_structured_full,
        }, f)
    print("SUCCESS: Candidate saved to artifacts/fully_gated_candidate.pkl")
