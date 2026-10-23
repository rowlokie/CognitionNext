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


# Polarity-paired terms: (reply_positive, article_negative) -> contradiction indicator
# Each pair: if reply contains the left word and article contains the right word -> suspicious
_POLARITY_PAIRS = [
    # coverage vs exclusion
    ("cover",    "exclud"),
    ("covered",  "exclud"),
    ("covers",   "exclud"),
    ("includ",   "exclud"),
    ("eligible", "not eligible"),
    ("eligible", "ineligible"),
    # reversal of allowed/prohibited
    ("allow",    "not allow"),
    ("allow",    "prohibit"),
    ("can ",     " cannot"),
    ("refundable", "non-refundable"),
    ("refundable", "nonrefundable"),
    ("transferable", "non-transferable"),
    ("transferable", "nontransferable"),
    # guaranteed vs conditional/optional
    ("guaranteed", "optional"),
    ("guaranteed", "not guaranteed"),
    ("included",   "optional"),
    ("included",   "not included"),
    ("included",   "add-on"),
    ("free",       "fee"),
    ("zero deductible", "deductible"),
    ("no deductible",   "deductible"),
    # Standard policy plan & role scope contradictions
    ("annual subscribers", "monthly subscribers"),
    ("monthly subscribers", "annual subscribers"),
    ("all users", "administrators"),
    ("administrators", "all users"),
    ("workspace owners", "all users"),
    ("all users", "workspace owners"),
    ("mandatory", "optional"),
    ("required", "optional"),
    ("standard", "optional add-on"),
]

WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "half": 0.5, "quarter": 0.25, "single": 1, "double": 2, "triple": 3,
}

_NUM_WORDS_STR = r"(?:half|quarter|single|double|triple|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:[\s\-]+(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand))*"
_NUM_STR = rf"(?:\d+(?:[.,]\d+)?|{_NUM_WORDS_STR}|a|an)"

_MONEY_RE = re.compile(
    rf"(?:\$\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?|{_NUM_WORDS_STR})\s*(?:dollars?|usd|\$))",
    re.IGNORECASE
)

_PERCENT_RE = re.compile(
    rf"({_NUM_STR})\s*(?:%|percent|pct)",
    re.IGNORECASE
)

_UNIT_STR = (
    r"(?:calendar\s+days?|business\s+days?|working\s+days?|days?|weeks?|months?|years?|hours?|minutes?|seconds?|"
    r"requests?\s+per\s+(?:second|minute|hour|day)|calls?\s+per\s+(?:second|minute|hour|day)|"
    r"failed\s+attempts?|attempts?|retries?|seats?|concurrent\s+jobs?|licenses?|devices?|"
    r"characters?|chars?|members?|users?)"
)

_DURATION_RE = re.compile(
    rf"\b({_NUM_STR})[\s-]+(?:(?:invited|active|total|max|maximum|concurrent|registered)\s+)?({_UNIT_STR})",
    re.IGNORECASE
)

# Negation words
_NEG_RE = re.compile(
    r"\b(not|no|never|cannot|can't|won't|don't|doesn't|isn't|aren't|weren't|"
    r"exclude|excluded|excludes|excluding|non-|ineligible|prohibited|restrict|restrict)\b",
    re.IGNORECASE,
)

def _split_sentences(text: str):
    """Simple sentence splitter on .!? followed by whitespace or end."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 8]

def parse_num(val_str: str) -> float:
    s = val_str.lower().strip().replace(",", "")
    if not s or s in ("a", "an"):
        return 1.0
    try:
        return float(s)
    except ValueError:
        pass

    words = re.split(r"[\s\-]+", s)
    total = 0.0
    current = 0.0
    for w in words:
        if w in WORD_NUMS:
            val = float(WORD_NUMS[w])
            if val == 100.0:
                current = (current if current != 0 else 1.0) * 100.0
            elif val == 1000.0:
                total += (current if current != 0 else 1.0) * 1000.0
                current = 0.0
            else:
                current += val
        else:
            try:
                current += float(w)
            except ValueError:
                pass
    return total + current

def get_context_words(text: str, start: int, end: int, window: int = 50) -> set:
    s = max(0, start - window)
    e = min(len(text), end + window)
    snippet = text[s:e].lower()
    tokens = set(re.findall(r"\b[a-z]{3,}\b", snippet))
    STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "you", "your", "our", "are", "was", "will", "can", "may", "have", "has"}
    return tokens - STOPWORDS

def normalize_duration(val: float, unit_raw: str):
    u = unit_raw.lower()
    if "second" in u:
        return val / 3600.0, "hours"
    if "minute" in u:
        return val / 60.0, "hours"
    if "hour" in u:
        return val, "hours"
    if "day" in u:
        return val * 24.0, "hours"
    if "week" in u:
        return val * 168.0, "hours"
    if "month" in u:
        return val * 720.0, "hours"
    if "year" in u:
        return val * 8760.0, "hours"
    return val * 24.0, "hours"

def are_durations_equivalent(d1_val: float, d1_unit: str, d2_val: float, d2_unit: str) -> bool:
    if d1_unit != d2_unit:
        return False
    diff = abs(d1_val - d2_val)
    denom = max(d1_val, d2_val, 1e-4)
    return (diff / denom) < 0.10

def extract_entities(text: str):
    entities = []
    text_lc = text.lower()

    for m in _MONEY_RE.finditer(text_lc):
        val_str = m.group(1) or m.group(2)
        val = parse_num(val_str)
        ctx = get_context_words(text_lc, m.start(), m.end())
        entities.append({
            "type": "money",
            "val": val,
            "unit": "usd",
            "raw": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "context": ctx,
        })

    for m in _PERCENT_RE.finditer(text_lc):
        val = parse_num(m.group(1))
        ctx = get_context_words(text_lc, m.start(), m.end())
        entities.append({
            "type": "percent",
            "val": val,
            "unit": "%",
            "raw": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "context": ctx,
        })

    ARTICLE_TIME_UNITS = {'second', 'minute', 'hour', 'day', 'week', 'month', 'year'}
    for m in _DURATION_RE.finditer(text_lc):
        val_raw = m.group(1)
        unit_raw = re.sub(r"\s+", " ", m.group(2).lower())
        if val_raw in ("a", "an") and not any(u in unit_raw for u in ARTICLE_TIME_UNITS):
            continue
        val = parse_num(val_raw)
        ctx = get_context_words(text_lc, m.start(), m.end())

        if "request" in unit_raw or "call" in unit_raw:
            ent_type = "rate"
            norm_unit = "req_per_min"
            norm_val = val * 60.0 if "second" in unit_raw else (val / 60.0 if "hour" in unit_raw else val)
        elif any(d in unit_raw for d in ARTICLE_TIME_UNITS):
            ent_type = "duration"
            norm_val, norm_unit = normalize_duration(val, unit_raw)
        else:
            ent_type = "quantity"
            norm_val = val
            norm_unit = "attempt" if "attempt" in unit_raw else ("seat" if "seat" in unit_raw else ("job" if "job" in unit_raw else "quota"))

        entities.append({
            "type": ent_type,
            "val": norm_val,
            "unit": norm_unit,
            "raw": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "context": ctx,
        })

    return entities

def compare_entities(reply: str, article: str):
    r_ents = extract_entities(reply)
    a_ents = extract_entities(article)

    if not r_ents:
        return {
            "num_r_ents": 0.0,
            "num_exact_matches": 0.0,
            "num_contradictions": 0.0,
            "unsupported_claims": 0.0,
            "max_contradiction_severity": 0.0,
            "has_numeric_contradiction": 0.0,
        }

    pairs = []
    for r_idx, re_ in enumerate(r_ents):
        for a_idx, ae in enumerate(a_ents):
            if re_["type"] == ae["type"] and re_["unit"] == ae["unit"]:
                common = re_["context"] & ae["context"]
                score = len(common)
                KEY_MATCHES = {"warranty", "return", "refund", "fee", "cost", "deductible", "cancel", "shipping", "replacement", "process", "lock", "expire", "reset", "clear"}
                score += len(common & KEY_MATCHES) * 3

                if re_["type"] == "duration":
                    eq = are_durations_equivalent(re_["val"], re_["unit"], ae["val"], ae["unit"])
                else:
                    eq = abs(re_["val"] - ae["val"]) < 1e-4

                if eq:
                    score += 15.0

                if score >= 1:
                    pairs.append((score, r_idx, a_idx))

    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_r = set()
    matched_a = set()

    num_exact_matches = 0
    num_contradictions = 0
    max_sev = 0.0

    for score, r_idx, a_idx in pairs:
        if r_idx in matched_r or a_idx in matched_a:
            continue
        matched_r.add(r_idx)
        matched_a.add(a_idx)
        re_ = r_ents[r_idx]
        ae = a_ents[a_idx]

        if re_["type"] == "duration":
            eq = are_durations_equivalent(re_["val"], re_["unit"], ae["val"], ae["unit"])
        else:
            eq = abs(re_["val"] - ae["val"]) < 1e-4

        if eq:
            num_exact_matches += 1
        else:
            num_contradictions += 1
            diff = abs(re_["val"] - ae["val"])
            denom = max(abs(ae["val"]), 1.0)
            max_sev = max(max_sev, min(diff / denom, 5.0))

    unsupported = 0
    art_words = set(re.findall(r"\b[a-z]{3,}\b", article.lower()))
    for r_idx, re_ in enumerate(r_ents):
        if r_idx not in matched_r:
            KEY_TOPICS = {"fee", "deductible", "cost", "warranty", "return", "refund", "discount"}
            if re_["context"] & KEY_TOPICS & art_words:
                unsupported += 1

    return {
        "num_r_ents": float(len(r_ents)),
        "num_exact_matches": float(num_exact_matches),
        "num_contradictions": float(num_contradictions),
        "unsupported_claims": float(unsupported),
        "max_contradiction_severity": float(max_sev),
        "has_numeric_contradiction": float(num_contradictions > 0),
    }

def enriched_grounding_features(reply: str, article: str) -> np.ndarray:
    """
    Comprehensive, calibrated 27-feature grounding representation:
    - Contextual numeric entity matching (money, duration, window, %)
    - Unit normalizations & semantic equivalence
    - Bipartite entity alignment with value agreement bonus
    - Contradiction-gated lexical overlap
    - Scaled contradiction penalties
    - Optional-to-entitlement conflict detection
    """
    r_lc = reply.lower()
    a_lc = article.lower()

    # Extract clean alphanumeric tokens (stripping trailing punctuation like . , ! ?)
    r_toks = re.findall(r"\b[a-z0-9]+\b", r_lc)
    a_toks = re.findall(r"\b[a-z0-9]+\b", a_lc)
    r_set = set(r_toks)
    a_set = set(a_toks)

    # 1. Base lexical overlap
    inter = r_set & a_set
    union = r_set | a_set
    jaccard       = len(inter) / len(union) if union else 0.0
    coverage_r    = len(inter) / len(r_set) if r_set else 0.0
    coverage_a    = len(inter) / len(a_set) if a_set else 0.0
    len_ratio_log = math.log1p(len(r_toks) / max(len(a_toks), 1))

    # 2. Content word precision/recall
    STOPWORDS = {"a","an","the","is","are","was","were","be","been","being",
                 "in","on","at","to","for","of","and","or","but","with",
                 "as","by","it","its","this","that","your","our","we","you",
                 "i","my","have","has","will","would","can","may","if","any"}
    r_content = r_set - STOPWORDS
    a_content = a_set - STOPWORDS
    c_inter   = r_content & a_content
    prec_content = len(c_inter) / len(r_content) if r_content else 0.0
    rec_content  = len(c_inter) / len(a_content) if a_content else 0.0

    # 3. Polarity contradiction
    polarity_hits = 0
    for r_word, a_word in _POLARITY_PAIRS:
        r_clean = r_word.strip()
        r_pat = rf"(?<!not\s)(?<!non-)\b{re.escape(r_clean)}\b"
        if re.search(r_pat, r_lc) and (a_word in a_lc):
            polarity_hits += 1
    polarity_contra = min(float(polarity_hits) / 3.0, 1.0)

    # 4. Contextual numeric entity comparison
    num_comp = compare_entities(reply, article)
    has_num_contra = num_comp["has_numeric_contradiction"]
    num_contra     = num_comp["num_contradictions"]
    num_matches    = num_comp["num_exact_matches"]
    num_sev        = num_comp["max_contradiction_severity"]
    unsupported    = num_comp["unsupported_claims"]

    contra_flag = float(has_num_contra > 0 or polarity_contra > 0.3)
    contra_intensity = min(has_num_contra * 2.0 + num_sev + (2.0 if contra_flag > 0 else 0.0) + polarity_contra * 2.0, 3.0)

    # Contradiction gating: when contradiction exists, overlap is NOT positive evidence
    gate = max(0.0, 1.0 - contra_intensity)
    gated_jaccard    = jaccard * gate
    gated_coverage_r = coverage_r * gate
    gated_coverage_a = coverage_a * gate
    gated_prec_c     = prec_content * gate
    gated_rec_c      = rec_content * gate

    # Direct contradiction penalty features (scaled so regularized linear model can offset overlap)
    contra_penalty_1 = contra_flag * 30.0
    contra_penalty_2 = contra_intensity * 30.0
    numeric_mismatch_penalty = has_num_contra * 60.0
    short_reply_contra = float(len(r_toks) <= 12 and has_num_contra > 0) * 40.0

    match_reward = float(num_matches > 0 and contra_flag == 0) * 10.0

    # 5. Sentence-level evidence alignment
    r_sents = _split_sentences(reply)
    a_sents = _split_sentences(article)
    if r_sents and a_sents:
        sent_supports = []
        for rs in r_sents:
            rs_toks = set(re.findall(r"\b[a-z0-9]+\b", rs.lower())) - STOPWORDS
            if not rs_toks:
                continue
            best = 0.0
            for as_ in a_sents:
                as_toks = set(re.findall(r"\b[a-z0-9]+\b", as_.lower())) - STOPWORDS
                if as_toks:
                    best = max(best, len(rs_toks & as_toks) / len(rs_toks))
            sent_supports.append(best)
        min_sent_support  = (min(sent_supports) if sent_supports else 0.0) * gate
        mean_sent_support = (float(np.mean(sent_supports)) if sent_supports else 0.0) * gate
        unsupported_ratio = float(sum(1 for s in sent_supports if s < 0.20)) / len(sent_supports)
    else:
        min_sent_support  = 0.0
        mean_sent_support = 0.0
        unsupported_ratio = float(not r_sents)

    # 6. Optional/entitlement signal
    optional_in_article = float(any(w in a_lc for w in (
        "optional", "add-on", "addon", "enroll", "separately", "available for", "upgrade"
    )))
    entitlement_in_reply = float(any(w in r_lc for w in (
        "included", "guaranteed", "automatically", "have care", "since you have",
        "your plan includes", "your subscription includes", "mandatory", "required"
    )))
    optional_entitlement_conflict = float(optional_in_article and entitlement_in_reply)

    # 7. Negation discrepancy
    r_negs = len(_NEG_RE.findall(r_lc))
    a_negs = len(_NEG_RE.findall(a_lc))
    neg_count_diff = float(abs(r_negs - a_negs))
    article_neg_reply_positive = float(a_negs > 0 and r_negs == 0)

    # 8. Per-sentence article coverage
    if r_sents and a_sents:
        art_toks_content = a_set - STOPWORDS
        per_sent_cov = []
        for rs in r_sents:
            rs_c = set(re.findall(r"\b[a-z0-9]+\b", rs.lower())) - STOPWORDS
            cov = len(rs_c & art_toks_content) / len(rs_c) if (rs_c and art_toks_content) else 0.0
            per_sent_cov.append(cov)
        mean_per_sent_art_cov = (float(np.mean(per_sent_cov)) if per_sent_cov else 0.0) * gate
        min_per_sent_art_cov  = (float(min(per_sent_cov)) if per_sent_cov else 0.0) * gate
    else:
        mean_per_sent_art_cov = 0.0
        min_per_sent_art_cov  = 0.0

    features = np.array([
        gated_jaccard,
        gated_coverage_r,
        gated_coverage_a,
        len_ratio_log,
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
        neg_count_diff,
        article_neg_reply_positive,
        min_sent_support,
        mean_sent_support,
        unsupported_ratio,
        optional_entitlement_conflict,
        mean_per_sent_art_cov,
        min_per_sent_art_cov,
    ], dtype=np.float32)
    return features


def overlap_features(reply: str, article: str) -> np.ndarray:
    """Legacy 4-feature overlap (kept for backward compat only)."""
    return enriched_grounding_features(reply, article)[:4]


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
    """Returns enriched 33-feature vector: tier(3) + amount(7) + grounding(23)."""
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([
        tier_onehot(tier),
        amount_features(amount),
        enriched_grounding_features(reply, article),
    ])


# -----------------------------------------------------------------
# Baseline model
# -----------------------------------------------------------------

class BaselineModel:
    """
    TF-IDF (word + char) single shared representation backbone + structured features
    + calibrated LogisticRegression for each of the three tasks.
    One shared feature representation backbone with three task heads.
    """

    def __init__(self):
        self.route_model = None
        self.approve_model = None
        self.grounded_model = None

        # ONE shared representation backbone (word + char TF-IDF across all tasks)
        self.shared_tfidf_word = None
        self.shared_tfidf_char = None

        # Prediction decision thresholds
        self.prediction_thresholds = {
            "approve": 0.730,
            "grounded": 0.510,
        }

        # Abstention thresholds (set after calibration)
        self.abstention_thresholds = {
            "route": 0.687228,
            "approve": 0.769230,
            "grounded": 0.561569,
        }

    @property
    def thresholds(self):
        return self.abstention_thresholds

    @thresholds.setter
    def thresholds(self, val):
        self.abstention_thresholds = val

    @property
    def route_tfidf_word(self):
        return self.shared_tfidf_word

    @property
    def route_tfidf_char(self):
        return self.shared_tfidf_char

    @property
    def approve_tfidf_word(self):
        return self.shared_tfidf_word

    @property
    def approve_tfidf_char(self):
        return self.shared_tfidf_char

    @property
    def grounded_tfidf_word(self):
        return self.shared_tfidf_word

    @property
    def grounded_tfidf_char(self):
        return self.shared_tfidf_char

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

        # Collect text across all three tasks for the shared representation backbone
        route_texts = [serialize_route(r["state"]) for r in train_records]
        route_struct = [structured_route(r["state"]) for r in train_records]
        route_labels = [r["labels"]["route"] for r in train_records]

        approve_texts = [serialize_approve(r["state"]) for r in train_records]
        approve_struct = [structured_approve(r["state"]) for r in train_records]
        approve_labels = [int(r["labels"]["approve"]) for r in train_records]

        grounded_texts = [serialize_grounded(r["state"]) for r in train_records]
        grounded_struct = [structured_grounded(r["state"]) for r in train_records]
        grounded_labels = [int(r["labels"]["grounded"]) for r in train_records]

        all_texts = route_texts + approve_texts + grounded_texts
        self.shared_tfidf_word, self.shared_tfidf_char = self._build_tfidf_pair()
        print(f"  Fitting ONE shared representation backbone on {len(all_texts)} combined text instances...")
        self.shared_tfidf_word.fit(all_texts)
        self.shared_tfidf_char.fit(all_texts)

        X_route = self._combine_features(
            route_texts, route_struct,
            self.shared_tfidf_word, self.shared_tfidf_char, fit=False
        )
        print(f"  Route features: {X_route.shape}")

        X_approve = self._combine_features(
            approve_texts, approve_struct,
            self.shared_tfidf_word, self.shared_tfidf_char, fit=False
        )
        print(f"  Approve features: {X_approve.shape}")

        X_grounded = self._combine_features(
            grounded_texts, grounded_struct,
            self.shared_tfidf_word, self.shared_tfidf_char, fit=False
        )
        print(f"  Grounded features: {X_grounded.shape}")

        # Train classifiers
        print("Training classifiers...")
        lr_kwargs = dict(max_iter=1000, C=1.0, class_weight="balanced", solver="saga")

        # Route: multiclass (multinomial handled automatically by saga solver)
        base_route = LogisticRegression(**lr_kwargs)
        self.route_model = CalibratedClassifierCV(base_route, method="isotonic", cv=3)
        self.route_model.fit(X_route, route_labels)
        print("  [OK] Route model trained")

        # Approve: binary
        base_approve = LogisticRegression(**lr_kwargs)
        self.approve_model = CalibratedClassifierCV(base_approve, method="isotonic", cv=3)
        self.approve_model.fit(X_approve, approve_labels)
        print("  [OK] Approve model trained")

        # Grounded: binary
        base_grounded = LogisticRegression(C=2.0, max_iter=500, class_weight="balanced", solver="lbfgs")
        self.grounded_model = CalibratedClassifierCV(base_grounded, method="isotonic", cv=3)
        self.grounded_model.fit(X_grounded, grounded_labels)
        print("  [OK] Grounded model trained")

    def _predict_route(self, state: dict):
        text = serialize_route(state)
        struct = structured_route(state)
        X = self._combine_features([text], [struct], self.shared_tfidf_word, self.shared_tfidf_char)
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
        abstain = confidence < self.abstention_thresholds.get("route", 0.5)
        return value, prob_dict, confidence, abstain

    def _predict_binary(self, state: dict, task: str):
        if task == "approve":
            text = serialize_approve(state)
            struct = structured_approve(state)
            X = self._combine_features([text], [struct], self.shared_tfidf_word, self.shared_tfidf_char)
            probs = self.approve_model.predict_proba(X)[0]
            classes = list(self.approve_model.classes_)
        else:  # grounded
            text = serialize_grounded(state)
            struct = structured_grounded(state)
            X = self._combine_features([text], [struct], self.shared_tfidf_word, self.shared_tfidf_char)
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
        thresh = self.prediction_thresholds.get(task, 0.5)
        value = p_true >= thresh
        confidence = max(p_true, 1 - p_true)
        abstain = confidence < self.abstention_thresholds.get(task, 0.5)

        # Epistemic uncertainty / incomplete evidence detection
        if task == "approve":
            text_full = (safe_str(state.get("subject", "")) + " " + safe_str(state.get("body", ""))).lower()
            UNCERTAINTY_PATTERNS = [
                "not yet available", "not recorded", "has not confirmed", "have not confirmed",
                "has not finished", "not complete", "no criteria supplied", "no approver decision",
                "terms are unavailable", "photo inconclusive", "logs are not yet available",
                "not yet been verified", "pending review"
            ]
            if any(pat in text_full for pat in UNCERTAINTY_PATTERNS):
                abstain = True

        elif task == "grounded":
            art = safe_str(state.get("kb_article", "")).lower()
            GAP_PATTERNS = [
                "not stated", "not specified", "not mentioned", "is not stated", "are not stated"
            ]
            if any(gp in art for gp in GAP_PATTERNS):
                abstain = True

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
        print(f"  [OK] Baseline model saved to {path}/baseline_model.pkl")

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
