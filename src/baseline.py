#!/usr/bin/env python3
"""
Phase 1 Baseline: TF-IDF + LogisticRegression + Structured Features
One shared interface: predict(state, decision) -> full response dict
Three calibrated classifiers, one per task.
"""

import json
import re
import math
import unicodedata
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TIERS = ["free", "pro", "enterprise"]
ROUTE_CLASSES = ["billing", "technical", "account", "shipping", "general"]

# -----------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------

def nfkc_normalize(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    text = nfkc_normalize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


# -----------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------

def tier_onehot(tier: Optional[str]) -> np.ndarray:
    """One-hot encode customer tier. Unknown tier -> all zeros."""
    vec = np.zeros(len(TIERS), dtype=np.float32)
    if tier in TIERS:
        vec[TIERS.index(tier)] = 1.0
    return vec


def amount_features(amount: Optional[float]) -> np.ndarray:
    """Log1p amount + amount band one-hot (5 bands)."""
    if amount is None or amount != amount:  # NaN check
        amount = 0.0
    amount = float(amount)
    log_amt = math.log1p(max(amount, 0))
    # bands: 0, 0-20, 20-50, 50-100, 100-200, 200+
    bands = [
        float(amount == 0),
        float(0 < amount < 20),
        float(20 <= amount < 50),
        float(50 <= amount < 100),
        float(100 <= amount < 200),
        float(amount >= 200),
    ]
    return np.array([log_amt] + bands, dtype=np.float32)


def text_length_features(subject: str, body: str) -> np.ndarray:
    """Character and word length features."""
    subj_wc = len(subject.split())
    body_wc = len(body.split())
    return np.array([
        math.log1p(subj_wc),
        math.log1p(body_wc),
        math.log1p(len(subject)),
        math.log1p(len(body)),
    ], dtype=np.float32)


def overlap_features(reply: str, article: str) -> np.ndarray:
    """Lexical overlap features for grounded task."""
    reply_tokens = set(reply.lower().split())
    article_tokens = set(article.lower().split())
    if not reply_tokens and not article_tokens:
        return np.zeros(4, dtype=np.float32)
    # Jaccard
    inter = reply_tokens & article_tokens
    union = reply_tokens | article_tokens
    jaccard = len(inter) / len(union) if union else 0.0
    # Coverage: fraction of reply tokens in article
    coverage_r = len(inter) / len(reply_tokens) if reply_tokens else 0.0
    # Coverage: fraction of article tokens in reply
    coverage_a = len(inter) / len(article_tokens) if article_tokens else 0.0
    # Reply/article length ratio
    len_ratio = len(reply_tokens) / max(len(article_tokens), 1)
    return np.array([jaccard, coverage_r, coverage_a, math.log1p(len_ratio)], dtype=np.float32)


# -----------------------------------------------------------------
# Serializers for each task
# -----------------------------------------------------------------

def serialize_route(state: dict) -> str:
    """Task-tagged text for route classifier."""
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    tier = clean_text(safe_str(state.get("customer_tier", "")))
    amount = state.get("amount", 0) or 0
    return f"[ROUTE] {subject} [SEP] {body} [SEP] {tier} | {amount:.2f}"


def serialize_approve(state: dict) -> str:
    """Task-tagged text for approve classifier."""
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    tier = clean_text(safe_str(state.get("customer_tier", "")))
    amount = state.get("amount", 0) or 0
    return f"[APPROVE] {subject} [SEP] {body} [SEP] {tier} | {amount:.2f}"


def serialize_grounded(state: dict) -> str:
    """Task-tagged text for grounded classifier."""
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return f"[GROUNDED] {reply} [SEP] {article}"


# -----------------------------------------------------------------
# Structured feature builders per task
# -----------------------------------------------------------------

def structured_route(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        text_length_features(subject, body),
    ])


def structured_approve(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        text_length_features(subject, body),
    ])


def structured_grounded(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        overlap_features(reply, article),
    ])


# -----------------------------------------------------------------
# Baseline model
# -----------------------------------------------------------------

class BaselineModel:
    """
    TF-IDF (word + char) + structured features + calibrated LogisticRegression
    for each of the three tasks. One shared sklearn pipeline per task.
    """

    def __init__(self):
        self.route_model = None
        self.approve_model = None
        self.grounded_model = None

        # TF-IDF vectorizers (shared word, shared char)
        self.route_tfidf_word = None
        self.route_tfidf_char = None
        self.approve_tfidf_word = None
        self.approve_tfidf_char = None
        self.grounded_tfidf_word = None
        self.grounded_tfidf_char = None

        # Abstention thresholds (set after calibration)
        self.thresholds = {
            "route": 0.5,
            "approve": 0.75,
            "grounded": 0.5,
        }

    def _build_tfidf_pair(self):
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30000,
            sublinear_tf=True,
            min_df=3,
        )
        return word_vec, char_vec

    def _combine_features(self, texts, structured_feats, word_vec, char_vec, fit=False):
        if fit:
            X_word = word_vec.fit_transform(texts)
            X_char = char_vec.fit_transform(texts)
        else:
            X_word = word_vec.transform(texts)
            X_char = char_vec.transform(texts)
        X_struct = csr_matrix(np.array(structured_feats, dtype=np.float32))
        return hstack([X_word, X_char, X_struct])

    def fit(self, train_records, calib_records=None):
        """
        Train on train_records.
        calib_records: held-out slice for calibration (if None, use CV on train).
        """
        print("Building feature matrices...")

        # Route
        route_texts = [serialize_route(r["state"]) for r in train_records]
        route_struct = [structured_route(r["state"]) for r in train_records]
        route_labels = [r["labels"]["route"] for r in train_records]

        self.route_tfidf_word, self.route_tfidf_char = self._build_tfidf_pair()
        X_route = self._combine_features(
            route_texts, route_struct,
            self.route_tfidf_word, self.route_tfidf_char, fit=True
        )
        print(f"  Route features: {X_route.shape}")

        # Approve
        approve_texts = [serialize_approve(r["state"]) for r in train_records]
        approve_struct = [structured_approve(r["state"]) for r in train_records]
        approve_labels = [int(r["labels"]["approve"]) for r in train_records]

        self.approve_tfidf_word, self.approve_tfidf_char = self._build_tfidf_pair()
        X_approve = self._combine_features(
            approve_texts, approve_struct,
            self.approve_tfidf_word, self.approve_tfidf_char, fit=True
        )
        print(f"  Approve features: {X_approve.shape}")

        # Grounded
        grounded_texts = [serialize_grounded(r["state"]) for r in train_records]
        grounded_struct = [structured_grounded(r["state"]) for r in train_records]
        grounded_labels = [int(r["labels"]["grounded"]) for r in train_records]

        self.grounded_tfidf_word, self.grounded_tfidf_char = self._build_tfidf_pair()
        X_grounded = self._combine_features(
            grounded_texts, grounded_struct,
            self.grounded_tfidf_word, self.grounded_tfidf_char, fit=True
        )
        print(f"  Grounded features: {X_grounded.shape}")

        # Train classifiers
        print("Training classifiers...")
        lr_kwargs = dict(max_iter=1000, C=1.0, class_weight="balanced", solver="saga", n_jobs=-1)

        # Route: multiclass (multinomial handled automatically by saga solver)
        base_route = LogisticRegression(**lr_kwargs)
        self.route_model = CalibratedClassifierCV(base_route, method="isotonic", cv=3)
        self.route_model.fit(X_route, route_labels)
        print("  ✓ Route model trained")

        # Approve: binary
        base_approve = LogisticRegression(**lr_kwargs)
        self.approve_model = CalibratedClassifierCV(base_approve, method="isotonic", cv=3)
        self.approve_model.fit(X_approve, approve_labels)
        print("  ✓ Approve model trained")

        # Grounded: binary
        base_grounded = LogisticRegression(**lr_kwargs)
        self.grounded_model = CalibratedClassifierCV(base_grounded, method="isotonic", cv=3)
        self.grounded_model.fit(X_grounded, grounded_labels)
        print("  ✓ Grounded model trained")

    def _predict_route(self, state: dict):
        text = serialize_route(state)
        struct = structured_route(state)
        X = self._combine_features([text], [struct], self.route_tfidf_word, self.route_tfidf_char)
        probs = self.route_model.predict_proba(X)[0]
        classes = self.route_model.classes_
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        # Fill any missing classes
        for cls in ROUTE_CLASSES:
            if cls not in prob_dict:
                prob_dict[cls] = 0.0
        # Renormalize to sum to 1
        total = sum(prob_dict.values())
        if total > 0:
            prob_dict = {k: v / total for k, v in prob_dict.items()}
        value = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[value]
        abstain = confidence < self.thresholds["route"]
        return value, prob_dict, confidence, abstain

    def _predict_binary(self, state: dict, task: str):
        if task == "approve":
            text = serialize_approve(state)
            struct = structured_approve(state)
            X = self._combine_features([text], [struct], self.approve_tfidf_word, self.approve_tfidf_char)
            probs = self.approve_model.predict_proba(X)[0]
            classes = list(self.approve_model.classes_)
        else:  # grounded
            text = serialize_grounded(state)
            struct = structured_grounded(state)
            X = self._combine_features([text], [struct], self.grounded_tfidf_word, self.grounded_tfidf_char)
            probs = self.grounded_model.predict_proba(X)[0]
            classes = list(self.grounded_model.classes_)

        # classes may be [0, 1] or [False, True]
        if 1 in classes:
            p_true = float(probs[classes.index(1)])
        elif True in classes:
            p_true = float(probs[classes.index(True)])
        else:
            p_true = float(probs[-1])

        p_true = max(0.0, min(1.0, p_true))
        value = p_true >= 0.5
        confidence = max(p_true, 1 - p_true)
        abstain = confidence < self.thresholds[task]
        return value, p_true, confidence, abstain

    def predict(self, state: dict, decision: str, request_id: str = "") -> dict:
        """Main predict interface. Returns full response dict."""
        t0 = time.perf_counter()

        if decision == "route":
            value, probs, confidence, abstain = self._predict_route(state)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "request_id": request_id,
                "decision": decision,
                "type": "choice",
                "value": value,
                "probs": {k: round(probs.get(k, 0.0), 6) for k in ROUTE_CLASSES},
                "confidence": round(confidence, 6),
                "abstain": bool(abstain),
                "latency_ms": round(latency_ms, 3),
            }
        elif decision in ("approve", "grounded"):
            value, p_true, confidence, abstain = self._predict_binary(state, decision)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "request_id": request_id,
                "decision": decision,
                "type": "binary",
                "value": bool(value),
                "p_true": round(p_true, 6),
                "abstain": bool(abstain),
                "latency_ms": round(latency_ms, 3),
            }
        else:
            raise ValueError(f"Unknown decision: {decision!r}. Must be one of: route, approve, grounded")

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "baseline_model.pkl", "wb") as f:
            pickle.dump(self, f, protocol=5)
        print(f"  ✓ Baseline model saved to {path}/baseline_model.pkl")

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        path = Path(path)
        with open(path / "baseline_model.pkl", "rb") as f:
            return pickle.load(f)


# -----------------------------------------------------------------
# Training helper
# -----------------------------------------------------------------

def load_jsonl_safe(path: str):
    records = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Validate required fields
                if "state" not in obj or obj["state"] is None:
                    print(f"  [WARN] Row {i}: missing 'state'")
                    bad += 1
                    continue
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Row {i}: malformed JSON: {e}")
                bad += 1
    if bad:
        print(f"  [INFO] Skipped {bad} malformed rows")
    return records


if __name__ == "__main__":
    import sys
    print("Training Phase 1 Baseline Model...")

    train_path = ROOT / "data" / "train.jsonl"
    dev_path = ROOT / "data" / "dev.jsonl"

    print(f"Loading train data from {train_path}")
    train_records = load_jsonl_safe(str(train_path))
    print(f"  Loaded {len(train_records)} train records")

    dev_records = load_jsonl_safe(str(dev_path))
    print(f"  Loaded {len(dev_records)} dev records")

    # Reserve ~10% of train for calibration
    np.random.seed(42)
    n_total = len(train_records)
    n_calib = max(int(n_total * 0.1), 100)
    indices = np.random.permutation(n_total)
    calib_idx = set(indices[:n_calib])
    calib_records = [train_records[i] for i in range(n_total) if i in calib_idx]
    fit_records = [train_records[i] for i in range(n_total) if i not in calib_idx]
    print(f"  Train (for fitting): {len(fit_records)}, Calibration: {len(calib_records)}")

    model = BaselineModel()
    model.fit(fit_records)

    model.save(ARTIFACTS_DIR)

    # Quick dev evaluation
    print("\nEvaluating on dev set...")
    from collections import Counter

    correct_route = 0
    tp_approve, fp_approve, fn_approve, tn_approve = 0, 0, 0, 0
    tp_grounded, fp_grounded, fn_grounded, tn_grounded = 0, 0, 0, 0
    route_preds = []
    route_golds = []

    for r in dev_records:
        state = r.get("state", {}) or {}
        labels = r.get("labels", {}) or {}

        # Route
        result = model.predict(state, "route", r.get("request_id", ""))
        pred = result["value"]
        gold = labels.get("route")
        route_preds.append(pred)
        route_golds.append(gold)
        if pred == gold:
            correct_route += 1

        # Approve
        result = model.predict(state, "approve", r.get("request_id", ""))
        pred = result["value"]
        gold = bool(labels.get("approve"))
        if pred and gold: tp_approve += 1
        elif pred and not gold: fp_approve += 1
        elif not pred and gold: fn_approve += 1
        else: tn_approve += 1

        # Grounded
        result = model.predict(state, "grounded", r.get("request_id", ""))
        pred = result["value"]
        gold = bool(labels.get("grounded"))
        if pred and gold: tp_grounded += 1
        elif pred and not gold: fp_grounded += 1
        elif not pred and gold: fn_grounded += 1
        else: tn_grounded += 1

    # Macro F1 for route
    classes = ROUTE_CLASSES
    f1s = []
    for cls in classes:
        tp = sum(1 for p, g in zip(route_preds, route_golds) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(route_preds, route_golds) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(route_preds, route_golds) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    macro_f1_route = sum(f1s) / len(f1s)

    # F1 for approve
    prec_a = tp_approve / (tp_approve + fp_approve) if (tp_approve + fp_approve) > 0 else 0
    rec_a = tp_approve / (tp_approve + fn_approve) if (tp_approve + fn_approve) > 0 else 0
    f1_approve = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0
    cost_approve = 3 * fp_approve + 1 * fn_approve

    # F1 for grounded
    prec_g = tp_grounded / (tp_grounded + fp_grounded) if (tp_grounded + fp_grounded) > 0 else 0
    rec_g = tp_grounded / (tp_grounded + fn_grounded) if (tp_grounded + fn_grounded) > 0 else 0
    f1_grounded = 2 * prec_g * rec_g / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0

    print("\n" + "="*55)
    print("  PHASE 1 BASELINE — DEV METRICS")
    print("="*55)
    print(f"  {'Task':<12} {'Metric':<25} {'Value':>8}")
    print("-"*55)
    print(f"  {'route':<12} {'macro-F1':<25} {macro_f1_route:>8.4f}")
    print(f"  {'approve':<12} {'F1':<25} {f1_approve:>8.4f}")
    print(f"  {'approve':<12} {'3*FP+1*FN (cost)':<25} {cost_approve:>8d}")
    print(f"  {'grounded':<12} {'macro-F1 (binary)':<25} {f1_grounded:>8.4f}")
    print("="*55)
