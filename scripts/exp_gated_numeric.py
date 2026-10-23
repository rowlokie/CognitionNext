"""
Test gated overlap + contradiction features on dev set and user test cases.
"""
import sys, json, pickle
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

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

def gated_grounded_features(reply: str, article: str) -> np.ndarray:
    base_f = enriched_grounding_features(reply, article)
    # base_f indices:
    # 0: jaccard, 1: coverage_r, 2: coverage_a, 3: len_ratio_log
    # 4: prec_content, 5: rec_content, 6: polarity_contra
    polarity_contra = base_f[6]

    num_comp = compare_entities_refined(reply, article)
    has_num_contra = num_comp["has_numeric_contradiction"]
    num_contra     = num_comp["num_contradictions"]
    num_matches    = num_comp["num_exact_matches"]
    num_sev        = num_comp["max_contradiction_severity"]
    unsupported    = num_comp["unsupported_claims"]

    # Contradiction flag & intensity:
    # 1.0 if direct numeric or polarity contradiction exists
    contra_flag = float(has_num_contra > 0 or polarity_contra > 0.3)
    contra_intensity = min(has_num_contra * 2.0 + num_sev + polarity_contra * 2.0, 3.0)

    # Gated overlap: when contradiction exists, overlap is NOT positive evidence
    gate = max(0.0, 1.0 - contra_intensity)
    gated_jaccard    = base_f[0] * gate
    gated_coverage_r = base_f[1] * gate
    gated_prec_c     = base_f[4] * gate
    gated_rec_c      = base_f[5] * gate

    # Direct contradiction penalty features (scaled so linear model has sufficient range)
    contra_penalty_1 = contra_flag * 25.0
    contra_penalty_2 = contra_intensity * 25.0
    numeric_mismatch_penalty = has_num_contra * 50.0
    short_reply_contra = float(len(reply.split()) <= 12 and has_num_contra > 0) * 30.0

    # Match reward
    match_reward = float(num_matches > 0 and contra_flag == 0) * 5.0

    features = np.array([
        # Base features (filtered/gated)
        gated_jaccard,
        gated_coverage_r,
        base_f[2],  # coverage_a
        base_f[3],  # len_ratio
        gated_prec_c,
        gated_rec_c,
        polarity_contra,
        # Numeric entities
        num_matches,
        num_contra,
        unsupported,
        num_sev,
        has_num_contra,
        # Gating and penalty features
        contra_flag,
        contra_intensity,
        contra_penalty_1,
        contra_penalty_2,
        numeric_mismatch_penalty,
        short_reply_contra,
        match_reward,
        # Additional base features
        base_f[10], # neg_count_diff
        base_f[11], # article_neg_reply_positive
        base_f[12] * gate, # min_sent_support
        base_f[13] * gate, # mean_sent_support
        base_f[14], # unsupported_ratio
        base_f[16], # optional_entitlement_conflict
    ], dtype=np.float32)

    return features

def build_structured_gated(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        gated_grounded_features(reply, article),
    ])

print("Building feature matrices...")
fit_texts   = [serialize_grounded(r["state"]) for r in fit_records]
fit_structs = [build_structured_gated(r["state"]) for r in fit_records]
y_fit       = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])

X_fit = hstack([
    shared_word.transform(fit_texts),
    shared_char.transform(fit_texts),
    csr_matrix(np.array(fit_structs, dtype=np.float32)),
])

print("Training CalibratedClassifierCV with gated features...")
base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", solver="lbfgs")
cand_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
cand_clf.fit(X_fit, y_fit)

dev_texts   = [serialize_grounded(r["state"]) for r in dev_records]
dev_structs = [build_structured_gated(r["state"]) for r in dev_records]
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

cand_preds = (dev_probs >= best_th).astype(int)
prec1 = precision_score(y_dev, cand_preds, pos_label=1)
rec1  = recall_score(y_dev, cand_preds, pos_label=1)
prec0 = precision_score(y_dev, cand_preds, pos_label=0)
rec0  = recall_score(y_dev, cand_preds, pos_label=0)

print(f"\nDev Macro-F1: {best_f1*100:.2f}% at threshold={best_th:.3f}")
print(f"Grounded (Class 1) Precision: {prec1*100:.2f}%, Recall: {rec1*100:.2f}%")
print(f"Ungrounded (Class 0) Precision: {prec0*100:.2f}%, Recall: {rec0*100:.2f}%")
print(f"Confusion Matrix:\n{confusion_matrix(y_dev, cand_preds)}")

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
print("Evaluating user cases with gated candidate model:")
print("=" * 65)
print(f"{'ID':<8} {'Expected':<10} {'Predicted':<10} {'p_true':<10} {'Status'}")
print("-" * 65)
all_pass = True
for tid, exp, kb, reply in TESTS:
    state = {"customer_tier": "pro", "amount": 0.0, "draft_reply": reply, "kb_article": kb}
    t = serialize_grounded(state)
    s = build_structured_gated(state)
    x = hstack([shared_word.transform([t]), shared_char.transform([t]), csr_matrix([s])])
    p = cand_clf.predict_proba(x)[0, 1]
    pred = p >= best_th
    status = "PASS" if pred == exp else "FAIL"
    if status == "FAIL": all_pass = False
    print(f"{tid:<8} {str(exp):<10} {str(pred):<10} {p:<10.4f} {status}")

print("=" * 65)
print(f"All 9 user test cases PASS: {all_pass}")

# Save candidate model to disk
with open(ROOT / "artifacts" / "gated_numeric_candidate.pkl", "wb") as f:
    pickle.dump({
        "grounded_model": cand_clf,
        "threshold": best_th,
        "macro_f1": best_f1,
    }, f)
print("Candidate saved to artifacts/gated_numeric_candidate.pkl")
