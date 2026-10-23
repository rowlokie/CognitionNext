"""
Prototype for contextual numeric entity extraction and contradiction detection.
"""
import re
from typing import List, Dict, Any, Tuple

# Word-to-number mapping
WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-four": 24, "twenty four": 24, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "half": 0.5, "half an": 0.5, "a": 1, "an": 1,
}

# Unit normalizations
# Duration to days or months or hours
DURATION_MULTIPLIERS = {
    "day": ("days", 1.0),
    "days": ("days", 1.0),
    "calendar day": ("days", 1.0),
    "calendar days": ("days", 1.0),
    "business day": ("business_days", 1.0),
    "business days": ("business_days", 1.0),
    "working day": ("business_days", 1.0),
    "working days": ("business_days", 1.0),
    "week": ("days", 7.0),
    "weeks": ("days", 7.0),
    "month": ("months", 1.0),
    "months": ("months", 1.0),
    "year": ("months", 12.0),
    "years": ("months", 12.0),
    "hour": ("hours", 1.0),
    "hours": ("hours", 1.0),
    "minute": ("hours", 1.0 / 60.0),
    "minutes": ("hours", 1.0 / 60.0),
}

# Regex for numbers: digits or number words
_NUM_WORDS_STR = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-four|twenty\s+four|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half|half\s+an|a|an)"
_NUM_STR = rf"(?:\d+(?:[.,]\d+)?|{_NUM_WORDS_STR})"

# Money regex: $100, 100 dollars, 100 usd
_MONEY_RE = re.compile(
    rf"(?:\$\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?|{_NUM_WORDS_STR})\s*(?:dollars?|usd|\$))",
    re.IGNORECASE
)

# Percent regex: 20%, 20 percent
_PERCENT_RE = re.compile(
    rf"({_NUM_STR})\s*(?:%|percent|pct)",
    re.IGNORECASE
)

# Duration regex: 14 calendar days, 30 days, 2 weeks, 12 months, 1 year, half an hour
_UNIT_STR = r"(?:calendar\s+days?|business\s+days?|working\s+days?|days?|weeks?|months?|years?|hours?|minutes?)"
_DURATION_RE = re.compile(
    rf"({_NUM_STR})\s+({_UNIT_STR})",
    re.IGNORECASE
)

def parse_num(val_str: str) -> float:
    val_str = val_str.lower().strip()
    if val_str in WORD_NUMS:
        return float(WORD_NUMS[val_str])
    # replace comma with dot
    val_str = val_str.replace(",", "")
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def get_context_words(text: str, start: int, end: int, window: int = 40) -> set:
    s = max(0, start - window)
    e = min(len(text), end + window)
    snippet = text[s:e].lower()
    # clean to alphanumeric tokens
    tokens = set(re.findall(r"\b[a-z]{3,}\b", snippet))
    # remove stopwords & generic terms
    STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "you", "your", "our", "are", "was", "will", "can", "may", "have", "has"}
    return tokens - STOPWORDS

def extract_entities(text: str) -> List[Dict[str, Any]]:
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
        # avoid matching 'a' if it was part of money already
        if val_raw in ("a", "an") and any(e["type"] == "money" and abs(e["start"] - m.start()) < 10 for e in entities):
            continue
        val = parse_num(val_raw)
        norm_unit, multiplier = DURATION_MULTIPLIERS.get(unit_raw, ("days", 1.0))
        norm_val = val * multiplier
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

def compare_entities(reply: str, article: str) -> Dict[str, float]:
    r_ents = extract_entities(reply)
    a_ents = extract_entities(article)

    num_exact_matches = 0
    num_contradictions = 0
    unsupported_claims = 0
    max_contradiction_severity = 0.0

    KEY_TOPICS = {
        "refund", "return", "warranty", "fee", "cost", "deductible",
        "cancel", "shipping", "replacement", "delivery", "limit",
        "window", "period", "process", "expire", "valid", "subscription",
        "repair", "damage", "coverage", "cover", "tier", "price"
    }

    for re_ in r_ents:
        re_type = re_["type"]
        re_unit = re_["unit"]
        re_val = re_["val"]
        re_ctx = re_["context"]
        re_topics = re_ctx & KEY_TOPICS

        # Look for matching entity in article
        best_match = None
        best_overlap = 0

        for ae in a_ents:
            if ae["type"] == re_type and ae["unit"] == re_unit:
                # compute context overlap
                common = re_ctx & ae["context"]
                common_topics = common & KEY_TOPICS
                score = len(common) + len(common_topics) * 2
                if score > best_overlap:
                    best_overlap = score
                    best_match = ae

        if best_match is not None and best_overlap >= 1:
            # We found a topical counterpart in the article!
            if abs(re_val - best_match["val"]) < 1e-4:
                num_exact_matches += 1
            else:
                num_contradictions += 1
                diff = abs(re_val - best_match["val"])
                denom = max(abs(best_match["val"]), 1.0)
                sev = min(diff / denom, 5.0)
                max_contradiction_severity = max(max_contradiction_severity, sev)
        else:
            # No matching entity with shared context found
            # Check if the topic of the claim exists in article
            art_words = set(re.findall(r"\b[a-z]{3,}\b", article.lower()))
            if re_topics & art_words:
                unsupported_claims += 1

    return {
        "num_r_ents": float(len(r_ents)),
        "num_a_ents": float(len(a_ents)),
        "num_exact_matches": float(num_exact_matches),
        "num_contradictions": float(num_contradictions),
        "unsupported_claims": float(unsupported_claims),
        "max_contradiction_severity": float(max_contradiction_severity),
        "has_numeric_contradiction": float(num_contradictions > 0),
    }

if __name__ == "__main__":
    cases = [
        ("Test 7: Return window", "Returns allowed within 30 days with no exceptions.", "Returns allowed within 14 calendar days, subject to conditions and exclusions."),
        ("Test 8: Refund deadline", "Refund processed within 30 calendar days.", "Approved refunds processed within 7 calendar days after approval; bank posting times may vary."),
        ("Test 9: Supported refund", "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.", "Approved refunds processed within 7 calendar days after approval; bank posting times may vary."),
        ("Test 10: Warranty duration", "Standard warranty lasts 24 months.", "Standard warranty lasts 12 months."),
        ("Test 11: Supported warranty", "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions.", "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions."),
        ("Test 13: Replacement fee", "Replacement fee is $250.", "Replacement fee is $100."),
    ]
    for name, r, a in cases:
        res = compare_entities(r, a)
        print(f"\n{name}:")
        print("  Reply:", r)
        print("  KB:   ", a)
        print("  Comparison:", res)
