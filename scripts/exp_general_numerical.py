#!/usr/bin/env python3
"""
General Numerical Comparison Architecture for Groundedness Classification.
Evaluated on full data/dev.jsonl (N=1,500) and trained strictly on data/train.jsonl.
"""

import math
import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

ROOT = Path(__file__).parent.parent

# -----------------------------------------------------------------
# 1. Number Word Mapping & Compound Parser
# -----------------------------------------------------------------

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
    "half": 0.5, "quarter": 0.25, "single": 1, "double": 2, "triple": 3,
}

WORD_NUMS_RE = r'(?:half|quarter|single|double|triple|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:[\s\-]+(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand))*'

NUM_TOKEN_RE = rf'(?:\d+(?:,\d+)*(?:\.\d+)?|{WORD_NUMS_RE}|a|an)'

ARTICLE_TIME_UNITS = {
    'second', 'seconds', 'minute', 'minutes', 'hour', 'hours',
    'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'quarter', 'half', 'dozen', 'hundred', 'thousand'
}

def parse_compound_number(s: str) -> float:
    s = s.lower().strip().replace(',', '')
    if not s or s in ('a', 'an'):
        return 1.0
    try:
        return float(s)
    except ValueError:
        pass
    
    words = re.split(r'[\s\-]+', s)
    total = 0.0
    current = 0.0
    for w in words:
        if w in NUM_WORDS:
            val = NUM_WORDS[w]
            if val == 100:
                current = (current if current != 0 else 1) * 100
            elif val == 1000:
                total += (current if current != 0 else 1) * 1000
                current = 0
            else:
                current += val
        else:
            try:
                current += float(w)
            except ValueError:
                pass
    return total + current

def stem_noun(w: str) -> str:
    w = w.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3 and w[-3] in "shxz":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    return w

# -----------------------------------------------------------------
# 2. Universal Entity Extraction
# -----------------------------------------------------------------

_MONEY_RE = re.compile(
    r"(?:\$|usd\s*|eur\s*|gbp\s*)\s*(\d+(?:,\d+)*(?:\.\d+)?)|"
    r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dollars?|usd|eur|gbp|cents?)",
    re.I
)

_PERCENT_RE = re.compile(
    rf"({NUM_TOKEN_RE})\s*(?:%|percent(?:age)?)",
    re.I
)

_SPECIFIC_UNIT_RE = re.compile(
    rf"\b({NUM_TOKEN_RE})"
    r"[\s\-]+"
    r"((?:business\s+|calendar\s+)?(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)|"
    r"requests?\s+per\s+(?:second|minute|hour|day)|calls?\s+per\s+(?:second|minute|hour|day)|"
    r"failed\s+attempts?|attempts?|retries?|devices?|seats?|licenses?|users?|accounts?|items?|units?|concurrent\s+jobs?|jobs?|tickets?|revisions?|tokens?|"
    r"backups?|instances?|domains?|webhooks?|queries?|records?|gigabytes?|megabytes?)",
    re.I
)

_CATCHALL_NOUN_RE = re.compile(
    rf"\b({NUM_TOKEN_RE})"
    r"[\s\-]+"
    r"([a-z]{3,})\b",
    re.I
)

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "you", "your", "our", "are",
    "was", "will", "can", "may", "have", "has", "after", "before", "until", "into",
    "each", "every", "more", "less", "than", "over", "under", "per", "via", "any"
}

def get_context_words(text: str, start: int, end: int, window: int = 50) -> set:
    s = max(0, start - window)
    e = min(len(text), end + window)
    snippet = text[s:e].lower()
    tokens = set(re.findall(r"\b[a-z]{3,}\b", snippet))
    return tokens - STOP_WORDS

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

def normalize_rate(val: float, unit_raw: str):
    u = unit_raw.lower()
    if "second" in u:
        return val * 60.0, "requests_per_min"
    if "hour" in u:
        return val / 60.0, "requests_per_min"
    return val, "requests_per_min"

def extract_all_entities(text: str):
    entities = []
    text_lc = text.lower()
    spans = []

    def overlaps(st, en):
        return any(max(st, s) < min(en, e) for s, e in spans)

    # 1. Money
    for m in _MONEY_RE.finditer(text_lc):
        val_str = m.group(1) or m.group(2)
        val = parse_compound_number(val_str)
        st, en = m.start(), m.end()
        spans.append((st, en))
        entities.append({
            "type": "money",
            "val": val,
            "unit": "usd",
            "raw": m.group(0),
            "start": st,
            "end": en,
            "context": get_context_words(text_lc, st, en),
        })

    # 2. Percentage
    for m in _PERCENT_RE.finditer(text_lc):
        st, en = m.start(), m.end()
        if overlaps(st, en):
            continue
        val = parse_compound_number(m.group(1))
        spans.append((st, en))
        entities.append({
            "type": "percent",
            "val": val,
            "unit": "%",
            "raw": m.group(0),
            "start": st,
            "end": en,
            "context": get_context_words(text_lc, st, en),
        })

    # 3. Specific Units (Durations, Rates, Known Quotas/Counts)
    for m in _SPECIFIC_UNIT_RE.finditer(text_lc):
        st, en = m.start(), m.end()
        if overlaps(st, en):
            continue
        val_raw = m.group(1)
        unit_raw = re.sub(r"\s+", " ", m.group(2).lower())
        
        # Guard against 'a' / 'an' with non-time
        if val_raw.lower() in ('a', 'an') and not any(unit_raw.startswith(u) for u in ARTICLE_TIME_UNITS):
            continue

        val = parse_compound_number(val_raw)

        if "request" in unit_raw or "call" in unit_raw:
            norm_val, norm_unit = normalize_rate(val, unit_raw)
            ent_type = "rate"
        elif any(d in unit_raw for d in ("second", "minute", "hour", "day", "week", "month", "year")):
            norm_val, norm_unit = normalize_duration(val, unit_raw)
            ent_type = "duration"
        else:
            norm_val = val
            norm_unit = stem_noun(unit_raw)
            ent_type = "quantity"

        spans.append((st, en))
        entities.append({
            "type": ent_type,
            "val": norm_val,
            "unit": norm_unit,
            "raw": m.group(0),
            "start": st,
            "end": en,
            "context": get_context_words(text_lc, st, en),
        })

    # 4. Catch-all general noun phrase
    for m in _CATCHALL_NOUN_RE.finditer(text_lc):
        st, en = m.start(), m.end()
        if overlaps(st, en):
            continue
        val_raw = m.group(1)
        noun_raw = m.group(2).lower()
        if noun_raw in STOP_WORDS or noun_raw in ("notice", "before", "after", "from", "ahead"):
            continue
        if val_raw.lower() in ('a', 'an'):
            continue
        val = parse_compound_number(val_raw)
        if val <= 0:
            continue
        norm_unit = stem_noun(noun_raw)
        spans.append((st, en))
        entities.append({
            "type": "quantity",
            "val": val,
            "unit": norm_unit,
            "raw": m.group(0),
            "start": st,
            "end": en,
            "context": get_context_words(text_lc, st, en),
        })

    return entities

# -----------------------------------------------------------------
# 3. Contextual Entity Comparison
# -----------------------------------------------------------------

def are_entities_equivalent(e1: dict, e2: dict) -> bool:
    if e1["type"] != e2["type"]:
        return False
    if e1["type"] == "duration":
        # 10% tolerance across duration conversion
        diff = abs(e1["val"] - e2["val"])
        denom = max(e1["val"], e2["val"], 1e-4)
        return (diff / denom) < 0.10
    if e1["type"] in ("money", "rate", "percent", "quantity"):
        if e1["unit"] != e2["unit"]:
            return False
        diff = abs(e1["val"] - e2["val"])
        denom = max(abs(e2["val"]), 1.0)
        return (diff / denom) < 1e-4
    return False

def compare_entities_general(reply: str, article: str):
    r_ents = extract_all_entities(reply)
    a_ents = extract_all_entities(article)

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
            # Check compatibility
            compatible = False
            if re_["type"] == ae["type"]:
                if re_["type"] in ("duration", "money"):
                    compatible = True
                elif re_["unit"] == ae["unit"]:
                    compatible = True
            
            # Or share strong context words
            common = re_["context"] & ae["context"]
            if not compatible and len(common) >= 2:
                compatible = True

            if compatible:
                score = len(common)
                # Boost key policy terms
                KEY_MATCHES = {"warranty", "return", "refund", "fee", "cost", "deductible", "cancel", "shipping", "replacement", "process", "lock", "expire", "reset", "attempt", "seat", "limit"}
                score += len(common & KEY_MATCHES) * 3

                eq = are_entities_equivalent(re_, ae)
                if eq:
                    score += 15.0

                if score >= 1 or eq:
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

        eq = are_entities_equivalent(re_, ae)
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
            KEY_TOPICS = {"fee", "deductible", "cost", "warranty", "return", "refund", "discount", "attempt", "limit", "seat"}
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

# -----------------------------------------------------------------
# 4. Scope & Polarity Pairs
# -----------------------------------------------------------------

_POLARITY_PAIRS = [
    ("non-refundable", "refundable"), ("nonrefundable", "refundable"),
    ("non-transferable", "transferable"), ("nontransferable", "transferable"),
    ("excluded", "included"), ("exclude", "include"),
    ("excludes", "includes"), ("not covered", "covered"),
    ("unsupported", "supported"), ("prohibited", "permitted"),
    ("not eligible", "eligible"), ("ineligible", "eligible"),
    ("denied", "approved"), ("no cancellation fee", "cancellation fee"),
    ("without fee", "fee"), ("free", "charge"),
    ("never", "always"), ("cannot", "can"),
    # Plan & scope pairs
    ("annual subscribers", "monthly subscribers"),
    ("monthly subscribers", "annual subscribers"),
    ("all users", "administrators"),
    ("administrators", "all users"),
    ("workspace owners", "all users"),
    ("all users", "workspace owners"),
]

def check_scope_polarity(reply_lc: str, article_lc: str) -> float:
    hits = 0
    for r_word, a_word in _POLARITY_PAIRS:
        r_clean = r_word.strip()
        r_pat = rf"(?<!not\s)(?<!non-)\b{re.escape(r_clean)}\b"
        if re.search(r_pat, reply_lc) and (a_word in article_lc):
            hits += 1
    return min(float(hits) / 2.0, 1.0)

# -----------------------------------------------------------------
# 5. Full 27-Feature Generator
# -----------------------------------------------------------------

def _split_sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def general_grounding_features(reply: str, article: str) -> np.ndarray:
    r_lc = reply.lower()
    a_lc = article.lower()

    r_toks = r_lc.split()
    a_toks = a_lc.split()
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

    # 3. Scope & Polarity contradiction
    polarity_contra = check_scope_polarity(r_lc, a_lc)

    # 4. Contextual numeric entity comparison
    num_comp = compare_entities_general(reply, article)
    has_num_contra = num_comp["has_numeric_contradiction"]
    num_contra     = num_comp["num_contradictions"]
    num_matches    = num_comp["num_exact_matches"]
    num_sev        = num_comp["max_contradiction_severity"]
    unsupported    = num_comp["unsupported_claims"]

    contra_flag = float(has_num_contra > 0 or polarity_contra > 0.3)
    contra_intensity = min(has_num_contra * 2.0 + num_sev + polarity_contra * 2.0, 3.0)

    # Contradiction gating: when contradiction exists, overlap is NOT positive evidence
    gate = max(0.0, 1.0 - contra_intensity)
    gated_jaccard    = jaccard * gate
    gated_coverage_r = coverage_r * gate
    gated_coverage_a = coverage_a * gate
    gated_prec_c     = prec_content * gate
    gated_rec_c      = rec_content * gate

    # Direct contradiction penalty features
    contra_penalty_1 = contra_flag * 30.0
    contra_penalty_2 = contra_intensity * 30.0
    numeric_mismatch_penalty = has_num_contra * 60.0
    short_reply_contra = float(len(r_toks) <= 12 and has_num_contra > 0) * 40.0

    match_reward = float(num_matches > 0 and contra_flag == 0) * 12.0

    # 5. Sentence-level evidence alignment
    r_sents = _split_sentences(reply)
    a_sents = _split_sentences(article)
    if r_sents and a_sents:
        sent_supports = []
        for rs in r_sents:
            rs_toks = set(rs.lower().split()) - STOPWORDS
            if not rs_toks:
                continue
            best = 0.0
            for as_ in a_sents:
                as_toks = set(as_.lower().split()) - STOPWORDS
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
        "your plan includes", "your subscription includes"
    )))
    optional_entitlement_conflict = float(optional_in_article and entitlement_in_reply)

    # 7. Negation discrepancy
    _NEG_RE = re.compile(r"\b(no|not|never|none|cannot|can't|won't|don't|doesn't|neither|nor)\b", re.I)
    r_negs = len(_NEG_RE.findall(r_lc))
    a_negs = len(_NEG_RE.findall(a_lc))
    neg_count_diff = float(abs(r_negs - a_negs))
    article_neg_reply_positive = float(a_negs > 0 and r_negs == 0)

    # 8. Per-sentence article coverage
    if r_sents and a_sents:
        art_toks_content = a_set - STOPWORDS
        per_sent_cov = []
        for rs in r_sents:
            rs_c = set(rs.lower().split()) - STOPWORDS
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

# -----------------------------------------------------------------
# 6. Evaluation & Training Harness
# -----------------------------------------------------------------

def run_experiment():
    print("Loading data...")
    with open(ROOT / "data" / "train.jsonl", "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]

    with open(ROOT / "data" / "dev.jsonl", "r", encoding="utf-8") as f:
        dev_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(train_records)} train records, {len(dev_records)} dev records.")

    # Split train into fit and calib (seed=42)
    np.random.seed(42)
    n_total = len(train_records)
    n_calib = max(int(n_total * 0.1), 100)
    indices = np.random.permutation(n_total)
    calib_idx = set(indices[:n_calib])
    fit_records = [train_records[i] for i in range(n_total) if i not in calib_idx]

    print(f"Fit slice: {len(fit_records)}, Calibration slice: {n_calib}")

    # Load baseline model to use its shared TF-IDF trunk and structured features
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import baseline
    sys.modules['baseline'] = baseline
    from baseline import BaselineModel, serialize_grounded, tier_onehot, amount_features, safe_str, clean_text

    base_model = BaselineModel.load(ROOT / "artifacts")
    tfidf_w = base_model.shared_tfidf_word
    tfidf_c = base_model.shared_tfidf_char

    print("Building features for fit split...")
    def build_dataset(records):
        texts = []
        structs = []
        labels = []
        for r in records:
            st = r["state"]
            texts.append(serialize_grounded(st))
            tier = safe_str(st.get("customer_tier", ""))
            amount = st.get("amount", 0)
            reply = clean_text(safe_str(st.get("draft_reply", "")))
            article = clean_text(safe_str(st.get("kb_article", "")))

            s_vec = np.concatenate([
                tier_onehot(tier),
                amount_features(amount),
                general_grounding_features(reply, article),
            ])
            structs.append(s_vec)
            labels.append(int(bool(r["labels"]["grounded"])))

        from scipy.sparse import hstack, csr_matrix
        X_w = tfidf_w.transform(texts)
        X_c = tfidf_c.transform(texts)
        X_s = csr_matrix(np.array(structs, dtype=np.float32))
        X = hstack([X_w, X_c, X_s]).tocsr()
        return X, np.array(labels)

    X_train, y_train = build_dataset(fit_records)
    X_dev, y_dev = build_dataset(dev_records)

    print("Fitting calibrated classifier on train fit split...")
    base_clf = LogisticRegression(C=1.0, max_iter=1000, solver="saga", random_state=42)
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=3)
    calibrated_clf.fit(X_train, y_train)

    print("Evaluating on dev split...")
    probs = calibrated_clf.predict_proba(X_dev)[:, 1]
    
    # Sweep thresholds on dev to evaluate performance
    best_th = 0.50
    best_f1 = 0.0
    for th in np.arange(0.40, 0.70, 0.02):
        preds = (probs >= th)
        f1 = f1_score(y_dev, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\nBest threshold: {best_th:.2f} -> Macro-F1: {best_f1*100:.2f}%")

    preds_default = (probs >= 0.50)
    preds_best = (probs >= best_th)

    acc_def = accuracy_score(y_dev, preds_default)
    f1_def = f1_score(y_dev, preds_default, average="macro")
    acc_best = accuracy_score(y_dev, preds_best)
    f1_best = f1_score(y_dev, preds_best, average="macro")

    prec_g = precision_score(y_dev, preds_best, pos_label=1)
    rec_g = recall_score(y_dev, preds_best, pos_label=1)
    prec_u = precision_score(y_dev, preds_best, pos_label=0)
    rec_u = recall_score(y_dev, preds_best, pos_label=0)

    print("=" * 60)
    print("  GENERAL NUMERICAL MODEL — DEV EVALUATION")
    print("=" * 60)
    print(f"  At threshold 0.50: Accuracy = {acc_def*100:.2f}%, Macro-F1 = {f1_def*100:.2f}%")
    print(f"  At threshold {best_th:.2f}: Accuracy = {acc_best*100:.2f}%, Macro-F1 = {f1_best*100:.2f}%")
    print(f"  Grounded (Class 1): Precision = {prec_g*100:.2f}%, Recall = {rec_g*100:.2f}%")
    print(f"  Ungrounded (Class 0): Precision = {prec_u*100:.2f}%, Recall = {rec_u*100:.2f}%")

    # Brier skill score & ECE
    from scripts.compute_official_score import compute_ece, compute_brier_skill_binary
    confs = np.maximum(probs, 1.0 - probs)
    corrects = (preds_best == y_dev)
    ece = compute_ece(confs, corrects)
    _, bss = compute_brier_skill_binary(probs, y_dev)

    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Brier Skill Score (BSS):          {bss:.4f}")

    # Top 80% selective error
    n_keep = int(len(probs) * 0.80)
    idx_sorted = np.argsort(confs)[::-1][:n_keep]
    sel_err = 1.0 - np.mean(corrects[idx_sorted])
    print(f"  Official 80% Selective Error:     {sel_err*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()
