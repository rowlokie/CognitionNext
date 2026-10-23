import sys
sys.path.insert(0, 'src')
import json, pickle, re, time, math
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix

from baseline import (
    load_jsonl_safe, clean_text, safe_str, tier_onehot, amount_features,
    serialize_grounded, structured_grounded
)

ROOT = Path(__file__).parent.parent

# -----------------------------------------------------------------------------
# Enriched Grounding Feature Extractor
# -----------------------------------------------------------------------------

NUM_PATTERN = re.compile(
    r"\b(?:\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%?|\d+\s*(?:days?|hours?|minutes?|weeks?|months?|years?|business\s+days?|seconds?))\b",
    re.IGNORECASE
)

NEGATION_WORDS = {"not", "no", "never", "cannot", "can't", "won't", "don't", "doesn't", "neither", "nor"}

def split_sentences(text: str):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

def extract_advanced_grounding_features(reply: str, article: str) -> np.ndarray:
    reply_clean = clean_text(reply).lower()
    art_clean = clean_text(article).lower()

    reply_tokens = reply_clean.split()
    art_tokens = art_clean.split()
    reply_set = set(reply_tokens)
    art_set = set(art_tokens)

    # 1. Base Lexical Overlap
    inter = reply_set & art_set
    union = reply_set | art_set
    jaccard = len(inter) / len(union) if union else 0.0
    prec_word = len(inter) / len(reply_set) if reply_set else 0.0  # reply precision
    rec_word = len(inter) / len(art_set) if art_set else 0.0      # reply coverage
    f1_word = 2 * prec_word * rec_word / (prec_word + rec_word) if (prec_word + rec_word) > 0 else 0.0
    len_ratio = len(reply_tokens) / max(len(art_tokens), 1)

    # 2. Number / Entity Mismatch Detection
    reply_nums = set(NUM_PATTERN.findall(reply_clean))
    art_nums = set(NUM_PATTERN.findall(art_clean))
    num_reply_nums = len(reply_nums)
    unmatched_nums = len(reply_nums - art_nums)
    match_nums_ratio = 1.0 if num_reply_nums == 0 else (len(reply_nums & art_nums) / num_reply_nums)
    has_unmatched_num = 1.0 if unmatched_nums > 0 else 0.0

    # 3. Negation & Contradiction Cues
    reply_neg = sum(1 for w in reply_tokens if w in NEGATION_WORDS)
    art_neg = sum(1 for w in art_tokens if w in NEGATION_WORDS)
    neg_diff = abs(reply_neg - art_neg)
    has_neg_conflict = 1.0 if (reply_neg > 0 and art_neg == 0) or (reply_neg == 0 and art_neg > 0) else 0.0

    # 4. Sentence-Level Evidence Alignment
    reply_sents = split_sentences(reply_clean)
    art_sents = split_sentences(art_clean)

    if reply_sents and art_sents:
        sent_max_overlaps = []
        for rs in reply_sents:
            rs_tokens = set(rs.split())
            if not rs_tokens:
                continue
            max_ov = 0.0
            for as_ in art_sents:
                as_tokens = set(as_.split())
                if as_tokens:
                    ov = len(rs_tokens & as_tokens) / len(rs_tokens)
                    if ov > max_ov:
                        max_ov = ov
            sent_max_overlaps.append(max_ov)

        min_sent_support = min(sent_max_overlaps) if sent_max_overlaps else 0.0
        mean_sent_support = float(np.mean(sent_max_overlaps)) if sent_max_overlaps else 0.0
        unsupported_sents = sum(1 for ov in sent_max_overlaps if ov < 0.25)
    else:
        min_sent_support = 0.0
        mean_sent_support = 0.0
        unsupported_sents = 0.0

    feats = [
        jaccard,
        prec_word,
        rec_word,
        f1_word,
        math.log1p(len_ratio),
        float(num_reply_nums),
        float(unmatched_nums),
        match_nums_ratio,
        has_unmatched_num,
        float(reply_neg),
        float(neg_diff),
        has_neg_conflict,
        min_sent_support,
        mean_sent_support,
        float(unsupported_sents),
    ]
    return np.array(feats, dtype=np.float32)

def structured_grounded_advanced(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        extract_advanced_grounding_features(reply, article),
    ])

def compute_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs <= hi) if lo == 0 else (probs > lo) & (probs <= hi)
        if mask.sum() > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)

def compute_brier_skill(probs, labels):
    base_rate = np.mean(labels)
    bs = np.mean((probs - labels) ** 2)
    bs_base = np.mean((base_rate - labels) ** 2)
    bss = 1.0 - (bs / (bs_base + 1e-12))
    return float(bs), float(bss)

def selective_error(probs, labels, threshold, coverage=0.80):
    confs = np.maximum(probs, 1.0 - probs)
    preds = (probs >= threshold).astype(int)
    errs = (preds != labels).astype(int)
    n_keep = int(len(confs) * coverage)
    idx = np.argsort(confs)[::-1][:n_keep]
    return float(np.mean(errs[idx]))

# -----------------------------------------------------------------------------
# Main Evaluation
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  PHASE 2: GROUNDED HALLUCINATION DETECTION IMPROVEMENT EXPERIMENT")
    print("=" * 70)

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

    # Load baseline model to extract its shared TF-IDF representation
    with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
        baseline_model = pickle.load(f)

    shared_word = baseline_model.shared_tfidf_word
    shared_char = baseline_model.shared_tfidf_char

    # 1. Evaluate Current Baseline on Dev
    t_start = time.perf_counter()
    base_probs = []
    golds = []
    for r in dev_records:
        res = baseline_model.predict(r["state"], "grounded")
        base_probs.append(res["p_true"])
        golds.append(int(bool(r["labels"]["grounded"])))
    base_lat_per_req = ((time.perf_counter() - t_start) / len(dev_records)) * 1000

    base_probs = np.array(base_probs)
    golds = np.array(golds)

    base_preds = (base_probs >= 0.510).astype(int)
    base_f1 = f1_score(golds, base_preds, average="macro")
    base_acc = accuracy_score(golds, base_preds)
    base_prec_0 = precision_score(golds, base_preds, pos_label=0)
    base_rec_0 = recall_score(golds, base_preds, pos_label=0)
    base_prec_1 = precision_score(golds, base_preds, pos_label=1)
    base_rec_1 = recall_score(golds, base_preds, pos_label=1)
    base_cm = confusion_matrix(golds, base_preds)
    base_ece = compute_ece(base_probs, golds)
    base_bs, base_bss = compute_brier_skill(base_probs, golds)
    base_sel_err = selective_error(base_probs, golds, 0.510, 0.80)

    # 2. Build & Train Candidate Grounded Model with Enriched Features
    print("\nExtracting enriched grounding features for fit set...")
    t_feat_start = time.perf_counter()
    fit_texts = [serialize_grounded(r["state"]) for r in fit_records]
    X_text_fit = baseline_model._combine_features(
        fit_texts,
        [structured_grounded_advanced(r["state"]) for r in fit_records],
        shared_word, shared_char, fit=False
    )
    y_fit = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])

    print("Training candidate Grounded head (Isotonic 3-fold CV)...")
    cand_base_clf = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", solver="saga")
    cand_clf = CalibratedClassifierCV(cand_base_clf, method="isotonic", cv=3)
    cand_clf.fit(X_text_fit, y_fit)

    # Transform Dev Set
    dev_texts = [serialize_grounded(r["state"]) for r in dev_records]
    t_cand_start = time.perf_counter()
    X_text_dev = baseline_model._combine_features(
        dev_texts,
        [structured_grounded_advanced(r["state"]) for r in dev_records],
        shared_word, shared_char, fit=False
    )
    cand_probs = cand_clf.predict_proba(X_text_dev)[:, 1]
    cand_lat_per_req = ((time.perf_counter() - t_cand_start) / len(dev_records)) * 1000

    # Tune threshold on dev for candidate
    best_th = 0.50
    best_f1 = -1
    for th in np.arange(0.30, 0.75, 0.01):
        p = (cand_probs >= th).astype(int)
        f = f1_score(golds, p, average="macro")
        if f > best_f1:
            best_f1 = f
            best_th = th

    cand_preds = (cand_probs >= best_th).astype(int)
    cand_acc = accuracy_score(golds, cand_preds)
    cand_prec_0 = precision_score(golds, cand_preds, pos_label=0)
    cand_rec_0 = recall_score(golds, cand_preds, pos_label=0)
    cand_prec_1 = precision_score(golds, cand_preds, pos_label=1)
    cand_rec_1 = recall_score(golds, cand_preds, pos_label=1)
    cand_cm = confusion_matrix(golds, cand_preds)
    cand_ece = compute_ece(cand_probs, golds)
    cand_bs, cand_bss = compute_brier_skill(cand_probs, golds)
    cand_sel_err = selective_error(cand_probs, golds, best_th, 0.80)

    # 3. Print Comparison Report
    print("\n" + "=" * 70)
    print("  GROUNDED MODEL COMPARISON: BASELINE vs ENRICHED CANDIDATE")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Baseline (Current)':>18} {'Candidate (Enriched)':>20} {'Diff':>10}")
    print("  " + "-" * 80)
    print(f"  {'Macro-F1':<30} {base_f1*100:>17.2f}% {best_f1*100:>19.2f}% {(best_f1-base_f1)*100:>+9.2f}%")
    print(f"  {'Accuracy':<30} {base_acc*100:>17.2f}% {cand_acc*100:>19.2f}% {(cand_acc-base_acc)*100:>+9.2f}%")
    print(f"  {'Class 0 (Ungrounded) Precision':<30} {base_prec_0*100:>17.2f}% {cand_prec_0*100:>19.2f}% {(cand_prec_0-base_prec_0)*100:>+9.2f}%")
    print(f"  {'Class 0 (Ungrounded) Recall':<30} {base_rec_0*100:>17.2f}% {cand_rec_0*100:>19.2f}% {(cand_rec_0-base_rec_0)*100:>+9.2f}%")
    print(f"  {'Class 1 (Grounded) Precision':<30} {base_prec_1*100:>17.2f}% {cand_prec_1*100:>19.2f}% {(cand_prec_1-base_prec_1)*100:>+9.2f}%")
    print(f"  {'Class 1 (Grounded) Recall':<30} {base_rec_1*100:>17.2f}% {cand_rec_1*100:>19.2f}% {(cand_rec_1-base_rec_1)*100:>+9.2f}%")
    print(f"  {'ECE (15 equal bins)':<30} {base_ece:>18.4f} {cand_ece:>20.4f} {cand_ece-base_ece:>+10.4f}")
    print(f"  {'Brier Skill Score':<30} {base_bss:>18.4f} {cand_bss:>20.4f} {cand_bss-base_bss:>+10.4f}")
    print(f"  {'Official 80% Selective Error':<30} {base_sel_err*100:>17.2f}% {cand_sel_err*100:>19.2f}% {(cand_sel_err-base_sel_err)*100:>+9.2f}%")
    print(f"  {'Inference Latency per req':<30} {base_lat_per_req:>16.2f}ms {cand_lat_per_req:>18.2f}ms {cand_lat_per_req-base_lat_per_req:>+8.2f}ms")

    print("\nConfusion Matrices:")
    print(f"  Baseline CM:\n{base_cm}")
    print(f"  Candidate CM (th={best_th:.2f}):\n{cand_cm}")

    # Save candidate artifact separately
    cand_artifact_path = ROOT / "artifacts" / "grounded_candidate.pkl"
    with open(cand_artifact_path, "wb") as f:
        pickle.dump({
            "model": cand_clf,
            "threshold": best_th,
            "macro_f1": best_f1,
            "ece": cand_ece,
            "selective_error": cand_sel_err
        }, f)
    print(f"\nSaved experimental candidate artifact to {cand_artifact_path}")

if __name__ == "__main__":
    main()
