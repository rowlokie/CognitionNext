"""
Refined contextual entity matching with:
1. Standard equivalence: 1 month == 30 days, 1 year == 12 months, 1 week == 7 days, 30 min == half an hour.
2. Bipartite optimal assignment matching (matching entities with highest semantic + syntactic overlap).
3. Attribute-specific anchoring (e.g., fee vs duration vs window).
"""
import re
from typing import List, Dict, Any, Tuple

WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-four": 24, "twenty four": 24, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "half": 0.5, "half an": 0.5, "a": 1, "an": 1,
}

_NUM_WORDS_STR = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-four|twenty\s+four|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half|half\s+an|a|an)"
_NUM_STR = rf"(?:\d+(?:[.,]\d+)?|{_NUM_WORDS_STR})"

_MONEY_RE = re.compile(
    rf"(?:\$\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?|{_NUM_WORDS_STR})\s*(?:dollars?|usd|\$))",
    re.IGNORECASE
)

_PERCENT_RE = re.compile(
    rf"({_NUM_STR})\s*(?:%|percent|pct)",
    re.IGNORECASE
)

_UNIT_STR = r"(?:calendar\s+days?|business\s+days?|working\s+days?|days?|weeks?|months?|years?|hours?|minutes?)"
_DURATION_RE = re.compile(
    rf"({_NUM_STR})\s+({_UNIT_STR})",
    re.IGNORECASE
)

def parse_num(val_str: str) -> float:
    val_str = val_str.lower().strip()
    if val_str in WORD_NUMS:
        return float(WORD_NUMS[val_str])
    val_str = val_str.replace(",", "")
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def get_context_words(text: str, start: int, end: int, window: int = 50) -> set:
    s = max(0, start - window)
    e = min(len(text), end + window)
    snippet = text[s:e].lower()
    tokens = set(re.findall(r"\b[a-z]{3,}\b", snippet))
    STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "you", "your", "our", "are", "was", "will", "can", "may", "have", "has"}
    return tokens - STOPWORDS

def normalize_duration(val: float, unit_raw: str) -> Tuple[float, str]:
    u = unit_raw.lower()
    if "business" in u or "working" in u:
        return val, "business_days"
    if "minute" in u:
        return val, "minutes"
    if "hour" in u:
        return val * 60.0, "minutes"
    if "day" in u:
        return val, "days"
    if "week" in u:
        return val * 7.0, "days"
    if "month" in u:
        # 1 month in days standard is 30 days
        return val * 30.0, "days"
    if "year" in u:
        return val * 365.0, "days"
    return val, "days"

def are_durations_equivalent(d1_val: float, d1_unit: str, d2_val: float, d2_unit: str) -> bool:
    if d1_unit != d2_unit:
        return False
    # If unit is days, allow 30 vs 31 days or exact
    if d1_unit == "days":
        # Check if 30 days == 1 month (30.0)
        diff = abs(d1_val - d2_val)
        if diff < 1e-3:
            return True
        # Allow 30 vs 31 days (calendar month)
        if (abs(d1_val - 30) < 1e-3 and abs(d2_val - 31) < 1e-3) or (abs(d2_val - 30) < 1e-3 and abs(d1_val - 31) < 1e-3):
            return True
        # Allow 365 vs 360 days (12 months vs 1 year)
        if (abs(d1_val - 365) < 1e-3 and abs(d2_val - 360) < 1e-3) or (abs(d2_val - 365) < 1e-3 and abs(d1_val - 360) < 1e-3):
            return True
        return False
    return abs(d1_val - d2_val) < 1e-3

def extract_entities_refined(text: str) -> List[Dict[str, Any]]:
    entities = []
    text_lc = text.lower()

    # 1. Money
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

    # 2. Percent
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

    # 3. Durations
    for m in _DURATION_RE.finditer(text_lc):
        val_raw = m.group(1)
        unit_raw = re.sub(r"\s+", " ", m.group(2).lower())
        if val_raw in ("a", "an") and any(e["type"] == "money" and abs(e["start"] - m.start()) < 10 for e in entities):
            continue
        val = parse_num(val_raw)
        norm_val, norm_unit = normalize_duration(val, unit_raw)
        ctx = get_context_words(text_lc, m.start(), m.end())
        entities.append({
            "type": "duration",
            "val": norm_val,
            "unit": norm_unit,
            "raw": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "context": ctx,
        })

    return entities

def compare_entities_refined(reply: str, article: str) -> Dict[str, float]:
    r_ents = extract_entities_refined(reply)
    a_ents = extract_entities_refined(article)

    if not r_ents:
        return {
            "num_r_ents": 0.0,
            "num_exact_matches": 0.0,
            "num_contradictions": 0.0,
            "unsupported_claims": 0.0,
            "max_contradiction_severity": 0.0,
            "has_numeric_contradiction": 0.0,
        }

    # Bipartite matching by context similarity
    pairs = []
    for r_idx, re_ in enumerate(r_ents):
        for a_idx, ae in enumerate(a_ents):
            if re_["type"] == ae["type"] and re_["unit"] == ae["unit"]:
                common = re_["context"] & ae["context"]
                score = len(common)
                # Boost if key noun/verb matches
                KEY_MATCHES = {"warranty", "return", "refund", "fee", "cost", "deductible", "cancel", "shipping", "replacement", "process", "lock", "expire", "reset", "clear"}
                score += len(common & KEY_MATCHES) * 3

                # Value agreement bonus: if values agree, strongly prefer matching them
                if re_["type"] == "duration":
                    eq = are_durations_equivalent(re_["val"], re_["unit"], ae["val"], ae["unit"])
                else:
                    eq = abs(re_["val"] - ae["val"]) < 1e-4

                if eq:
                    score += 15.0  # Agreement bonus!

                if score >= 1:
                    pairs.append((score, r_idx, a_idx))

    # Sort descending by score
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

    # Unmatched reply entities on topics mentioned in article
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

if __name__ == "__main__":
    # Test on the false positives from earlier
    cases = [
        ("FP1: thirty days vs a month",
         "all plans get thirty days to claim, so yours qualifies. I have sent it back to the card on file; expect it in up to 10 business days.",
         "Refunds may be requested within a month of the charge, and this applies to all plans. Claims are reviewed within 7 days. Approved refunds are returned to the card on file and typically settle in up to 10 business days."),
        ("FP2: a month vs thirty days",
         "Queued. all users get exports in CSV and JSON; the link is valid a month and it takes up to an hour to build. Related requests are actioned within six days.",
         "all users can export at any time. The download stays live thirty days and is delivered in CSV and JSON. Preparation needs up to an hour; up to 6 days concurrent jobs are allowed."),
        ("FP3: half an hour vs 30 minutes",
         "I have sent a reset by an email link, which is how administrators recover access. The link lasts half an hour, and the account locks after three failed attempts. Replacements and follow-ups are handled within ten minutes.",
         "Password resets are available to all users via an email link. Links expire after 30 minutes. Accounts lock automatically after three failed attempts and clear in 10 minutes minutes."),
        ("Test 8: 30 days vs 7 days",
         "Refund processed within 30 calendar days.",
         "Approved refunds processed within 7 calendar days after approval; bank posting times may vary."),
        ("Test 10: 24 months vs 12 months",
         "Standard warranty lasts 24 months.",
         "Standard warranty lasts 12 months."),
        ("Test 13: $250 vs $100",
         "Replacement fee is $250.",
         "Replacement fee is $100.")
    ]
    for name, r, a in cases:
        c = compare_entities_refined(r, a)
        print(f"\n{name}:")
        print("  has_numeric_contradiction:", c["has_numeric_contradiction"])
        print("  matches:", c["num_exact_matches"], "contradictions:", c["num_contradictions"])
