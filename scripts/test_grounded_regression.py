"""
Comprehensive category-stratified grounded regression test suite.
Evaluates:
  1. Numerical mismatches in prices, durations, deadlines, and quantities
  2. Correct replies with matching numbers
  3. Numbers irrelevant to the factual claim (order IDs, customer context)
  4. Unsupported numerical claims
  5. Semantic contradictions (coverage, exclusions, eligibility, guarantees)
  6. Correct paraphrases of knowledge-base articles
  7. Information gaps (KB silent on specific claim)
  8. Abstention behavior

Usage:
  python scripts/test_grounded_regression.py
"""
import sys, json, requests
from collections import defaultdict

URL = "http://127.0.0.1:8000/v1/decide"

REGRESSION_CASES = [
    # ----------------------------------------------------------------
    # 1. Numerical Mismatches: Prices, Durations, Deadlines, Quantities
    # ----------------------------------------------------------------
    {
        "id": "num-price-01", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Replacement fee mismatch: $250 claimed vs $100 in KB",
        "draft_reply": "Your replacement fee is $250.",
        "kb_article": "Replacement fee is $100.",
    },
    {
        "id": "num-price-02", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Refund limit mismatch: $200 claimed vs $100 cap",
        "draft_reply": "Our policy allows refunds of up to $200 per request.",
        "kb_article": "Refunds are capped at $100 per transaction. Amounts above this require approval.",
    },
    {
        "id": "num-dur-01", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Warranty duration mismatch: 24 months claimed vs 12 months in KB",
        "draft_reply": "Standard warranty lasts 24 months.",
        "kb_article": "Standard warranty lasts 12 months.",
    },
    {
        "id": "num-dur-02", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Warranty duration mismatch: 3-year warranty claimed vs 1-year in KB",
        "draft_reply": "Your device is covered by a 3-year manufacturer warranty.",
        "kb_article": "All devices come with a standard 1-year manufacturer warranty. Extended warranties can be purchased separately.",
    },
    {
        "id": "num-dead-01", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Refund deadline mismatch: 30 calendar days claimed vs 7 calendar days in KB",
        "draft_reply": "Refund processed within 30 calendar days.",
        "kb_article": "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.",
    },
    {
        "id": "num-dead-02", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Refund speed overstated: 24 hours claimed vs 5-7 business days in KB",
        "draft_reply": "Returns are accepted within 30 days, and your refund will be issued within 24 hours.",
        "kb_article": "Returns are accepted within 30 days of delivery. Approved refunds are processed within 5-7 business days.",
    },
    {
        "id": "num-win-01", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Return window contradiction: 30 days claimed vs 14 calendar days in KB",
        "draft_reply": "Returns allowed within 30 days with no exceptions.",
        "kb_article": "Returns allowed within 14 calendar days, subject to conditions and exclusions.",
    },
    {
        "id": "num-win-02", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Return window contradiction: 60 days claimed vs 30 days in KB",
        "draft_reply": "Items must be returned within 60 days of purchase.",
        "kb_article": "Returns are accepted within 30 days of delivery. Items must be in original condition.",
    },
    {
        "id": "num-qty-01", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Free shipping threshold: $30 claimed vs $75 in KB",
        "draft_reply": "Free shipping is available on orders over $30.",
        "kb_article": "Free shipping applies to orders totaling $75 or more before tax.",
    },
    {
        "id": "num-deduct-01", "category": "Numerical Mismatch", "expected": "UNGROUNDED",
        "desc": "Deductible mismatch: $25 claimed vs $75 in KB",
        "draft_reply": "Your claim will be processed with a $25 deductible.",
        "kb_article": "AccidentalCare claims are subject to a $75 deductible per incident.",
    },

    # ----------------------------------------------------------------
    # 2. Correct Replies with Matching Numbers & Equivalent Units
    # ----------------------------------------------------------------
    {
        "id": "match-dur-01", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Supported warranty duration: 12 months accurately repeated",
        "draft_reply": "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions.",
        "kb_article": "Standard warranty lasts 12 months and covers manufacturing defects subject to exclusions.",
    },
    {
        "id": "match-dur-02", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Supported warranty duration: 1-year matches 1-year in KB",
        "draft_reply": "All devices are covered by a standard 1-year manufacturer warranty.",
        "kb_article": "All devices come with a standard 1-year manufacturer warranty. Extended warranties can be purchased separately.",
    },
    {
        "id": "match-dur-03", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Equivalent duration: 1 month in draft matches 30 days in KB",
        "draft_reply": "The export link is valid a month and is delivered in CSV format.",
        "kb_article": "All users can export at any time. The download stays live thirty days and is delivered in CSV format.",
    },
    {
        "id": "match-dead-01", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Supported refund timeline: 7 calendar days accurately repeated",
        "draft_reply": "Approved refunds are processed within 7 calendar days after approval; bank posting times may vary.",
        "kb_article": "Approved refunds processed within 7 calendar days after approval; bank posting times may vary.",
    },
    {
        "id": "match-dead-02", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Equivalent time: half an hour in draft matches 30 minutes in KB",
        "draft_reply": "Your reset link lasts half an hour. Accounts lock after three failed attempts.",
        "kb_article": "Links expire after 30 minutes. Accounts lock automatically after three failed attempts.",
    },
    {
        "id": "match-fee-01", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Matching price: $5.99 shipping cost accurately stated",
        "draft_reply": "Shipping costs $5.99 for orders under $50. Orders over $50 qualify for free shipping.",
        "kb_article": "Shipping costs of $5.99 apply to all orders under $50. Free shipping is available for orders over $50.",
    },
    {
        "id": "match-fee-02", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Matching fee: $25 early termination fee accurately repeated",
        "draft_reply": "Cancelling within the first 3 months will incur a $25 early termination fee.",
        "kb_article": "Cancellation within the first 3 months incurs a $25 early termination fee. After 3 months, cancellation is free.",
    },
    {
        "id": "match-deduct-01", "category": "Matching Numbers", "expected": "GROUNDED",
        "desc": "Matching deductible: $75 deductible accurately repeated",
        "draft_reply": "AccidentalCare claims require a $75 deductible per incident.",
        "kb_article": "AccidentalCare claims are subject to a $75 deductible per incident.",
    },

    # ----------------------------------------------------------------
    # 3. Numbers Irrelevant to Factual Claim (Customer Context / State)
    # ----------------------------------------------------------------
    {
        "id": "irrel-01", "category": "Irrelevant Numbers", "expected": "GROUNDED",
        "desc": "Order amount $45 mentioned from customer ticket context, not contradicting KB",
        "draft_reply": "Your $45 order qualifies for a return under our 30-day policy. Please return in original condition.",
        "kb_article": "Returns are accepted within 30 days of delivery. Items must be in original condition.",
    },
    {
        "id": "irrel-02", "category": "Irrelevant Numbers", "expected": "GROUNDED",
        "desc": "Ticket ID #9823 mentioned in reply, not contradicting policy",
        "draft_reply": "Regarding ticket #9823, priority support is available for Pro tier customers.",
        "kb_article": "Priority support is available exclusively for Pro and Enterprise tier customers.",
    },
    {
        "id": "irrel-03", "category": "Irrelevant Numbers", "expected": "GROUNDED",
        "desc": "User count mentioned from customer setup, policy rules respected",
        "draft_reply": "For your team of 5 users, licenses are non-transferable between devices.",
        "kb_article": "Licenses are non-transferable and may only be used on the device originally registered.",
    },

    # ----------------------------------------------------------------
    # 4. Unsupported Numerical Claims (Numbers without KB basis)
    # ----------------------------------------------------------------
    {
        "id": "unsupp-01", "category": "Unsupported Numerical", "expected": "UNGROUNDED",
        "desc": "Zero deductible claimed on Care+ when KB does not state zero deductible",
        "draft_reply": "Care+ has a zero deductible.",
        "kb_article": "Care+ is optional; applicable charges are specified in the customer's policy.",
    },
    {
        "id": "unsupp-02", "category": "Unsupported Numerical", "expected": "UNGROUNDED",
        "desc": "20% discount promised when KB says nothing about discounts",
        "draft_reply": "As a loyal customer, you qualify for a 20% discount on your next renewal.",
        "kb_article": "Renewal pricing is based on the current plan rate at the time of renewal.",
    },
    {
        "id": "unsupp-03", "category": "Unsupported Numerical", "expected": "UNGROUNDED",
        "desc": "$50 credit promised when KB is silent",
        "draft_reply": "As a loyal customer you will receive a $50 credit applied to your account.",
        "kb_article": "We appreciate your continued business. Please contact our loyalty team for assistance.",
    },
    {
        "id": "unsupp-04", "category": "Unsupported Numerical", "expected": "UNGROUNDED",
        "desc": "Refund amount $150 and 24hr speed promised on empty review article",
        "draft_reply": "Your request is approved and a refund of $150 will be issued within 24 hours.",
        "kb_article": "Thank you for reaching out. We will review your request.",
    },

    # ----------------------------------------------------------------
    # 5. Semantic Contradictions (Coverage, Exclusions, Eligibility, Guarantees)
    # ----------------------------------------------------------------
    {
        "id": "sem-cov-01", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Warranty covers water damage — KB strictly excludes liquid ingress",
        "draft_reply": "Pro warranty covers water damage caused by full immersion in a swimming pool.",
        "kb_article": "Standard and Pro warranties exclude liquid ingress and submersion.",
    },
    {
        "id": "sem-cov-02", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Cosmetic damage covered — KB strictly excludes scratches and dents",
        "draft_reply": "Our warranty covers cosmetic damage including scratches and dents on the device.",
        "kb_article": "Warranty coverage excludes cosmetic damage such as scratches, dents, and broken ports not affecting functionality.",
    },
    {
        "id": "sem-guar-01", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Free replacement guaranteed regardless of cause — KB limits to manufacturing defects",
        "draft_reply": "Guarantees a free replacement for every damaged device regardless of cause.",
        "kb_article": "Free replacements are limited to confirmed manufacturing defects; accidental and physical damage are excluded.",
    },
    {
        "id": "sem-neg-01", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Accidental damage described as covered — KB says standard does not cover it",
        "draft_reply": "Accidental drops and spills are covered under your standard plan.",
        "kb_article": "Standard plans do not cover accidental damage. An AccidentalCare add-on is available for $9.99/month.",
    },
    {
        "id": "sem-neg-02", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Non-refundable fee described as refundable",
        "draft_reply": "The annual subscription fee is fully refundable if cancelled within 60 days.",
        "kb_article": "Annual subscription fees are non-refundable. Customers may cancel future renewals at any time.",
    },
    {
        "id": "sem-neg-03", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Non-transferable license described as transferable",
        "draft_reply": "You can transfer your license to another device at any time.",
        "kb_article": "Licenses are non-transferable and may only be used on the device originally registered.",
    },
    {
        "id": "sem-elig-01", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Free tier asserted eligible for priority support — KB says not eligible",
        "draft_reply": "Free-tier accounts are eligible for priority support.",
        "kb_article": "Priority support is available exclusively for Pro and Enterprise tier customers. Free-tier accounts are not eligible.",
    },
    {
        "id": "sem-opt-01", "category": "Semantic Contradiction", "expected": "UNGROUNDED",
        "desc": "Care+ assumed as included without customer enrollment",
        "draft_reply": "Since you have Care+, your liquid damage claim is fully covered.",
        "kb_article": "Accidental liquid damage is covered under the optional Care+ add-on. Customers must enroll separately.",
    },

    # ----------------------------------------------------------------
    # 6. Correct Paraphrases of Knowledge-Base Articles
    # ----------------------------------------------------------------
    {
        "id": "para-01", "category": "Correct Paraphrase", "expected": "GROUNDED",
        "desc": "Accurate paraphrase of warranty liquid exclusion",
        "draft_reply": "Liquid immersion and submersion are excluded from Standard and Pro warranties. Accidental liquid damage requires an optional Care+ policy.",
        "kb_article": "Standard and Pro warranties strictly exclude liquid ingress, submersion, and cosmetic damage. Accidental liquid damage requires an optional Care+ policy.",
    },
    {
        "id": "para-02", "category": "Correct Paraphrase", "expected": "GROUNDED",
        "desc": "Accurate paraphrase of priority support eligibility",
        "draft_reply": "Priority support is available exclusively for Pro and Enterprise tier customers. Free-tier accounts are not eligible.",
        "kb_article": "Priority support is available exclusively for Pro and Enterprise tier customers. Free-tier accounts are not eligible.",
    },
    {
        "id": "para-03", "category": "Correct Paraphrase", "expected": "GROUNDED",
        "desc": "Accurate non-transferable license paraphrase",
        "draft_reply": "Licenses are non-transferable and tied to the original registered device.",
        "kb_article": "Licenses are non-transferable and may only be used on the device originally registered.",
    },
    {
        "id": "para-04", "category": "Correct Paraphrase", "expected": "GROUNDED",
        "desc": "Accurate Care+ opt-in requirement paraphrase",
        "draft_reply": "Liquid damage is covered only if you have enrolled in the optional Care+ add-on.",
        "kb_article": "Accidental liquid damage is covered under the optional Care+ add-on. Customers must enroll separately.",
    },
    {
        "id": "para-05", "category": "Correct Paraphrase", "expected": "GROUNDED",
        "desc": "Support contact redirection without ungrounded claims",
        "draft_reply": "Please contact our billing support team for assistance with your disputed invoice.",
        "kb_article": "For billing disputes, please contact our billing support team at billing@example.com.",
    },

    # ----------------------------------------------------------------
    # 7. Information Gaps / Silent Article Cases
    # ----------------------------------------------------------------
    {
        "id": "gap-01", "category": "Information Gap", "expected": "UNGROUNDED",
        "desc": "Resolution time guaranteed when article only states response target",
        "draft_reply": "We guarantee this issue will be fully resolved within 24 hours.",
        "kb_article": "Our support team aims to respond within 2 business days.",
    },
    {
        "id": "gap-02", "category": "Information Gap", "expected": "UNGROUNDED",
        "desc": "Article silent on grandfathering — draft promises price freeze",
        "draft_reply": "Existing customers are grandfathered into the old pricing and will not be affected by price increases.",
        "kb_article": "Our pricing is subject to change. Customers will receive 30 days advance notice before any price changes take effect.",
    },
    {
        "id": "gap-03", "category": "Information Gap", "expected": "UNGROUNDED",
        "desc": "Article silent on expedited processing for enterprise",
        "draft_reply": "Enterprise customers receive expedited claim processing within 24 hours.",
        "kb_article": "Claims are processed in the order they are received. Standard processing time is 5-7 business days.",
    },
]

def run_regression():
    print("=" * 80)
    print("  GROUNDEDNESS REGRESSION TEST SUITE (Live API)")
    print("=" * 80)

    results = []
    cat_stats = defaultdict(lambda: {"total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})

    for c in REGRESSION_CASES:
        payload = {
            "request_id": c["id"],
            "decision": "grounded",
            "state": {
                "customer_tier": "pro",
                "amount": 0.0,
                "draft_reply": c["draft_reply"],
                "kb_article": c["kb_article"],
            }
        }
        r = requests.post(URL, json=payload, timeout=5).json()
        val = r["value"]
        p_true = r["p_true"]
        abstain = r["abstain"]
        pred = "GROUNDED" if val else "UNGROUNDED"
        correct = (pred == c["expected"])

        cat = c["category"]
        cat_stats[cat]["total"] += 1
        if correct:
            cat_stats[cat]["correct"] += 1

        if c["expected"] == "GROUNDED":
            if correct:
                cat_stats[cat]["tp"] += 1
            else:
                cat_stats[cat]["fn"] += 1
        else:
            if correct:
                cat_stats[cat]["tn"] += 1
            else:
                cat_stats[cat]["fp"] += 1

        results.append({
            "id": c["id"],
            "category": cat,
            "desc": c["desc"],
            "expected": c["expected"],
            "predicted": pred,
            "correct": correct,
            "p_true": p_true,
            "abstain": abstain,
        })

    # Print results table
    print(f"{'ID':<16} {'Category':<24} {'Expected':<12} {'Predicted':<12} {'p_true':<8} {'Status'}")
    print("-" * 80)
    for res in results:
        status = "[PASS]" if res["correct"] else "[FAIL] <<<<<"
        print(f"{res['id']:<16} {res['category']:<24} {res['expected']:<12} {res['predicted']:<12} {res['p_true']:<8.4f} {status}")

    # Summary calculations
    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    tp = sum(cs["tp"] for cs in cat_stats.values())
    fp = sum(cs["fp"] for cs in cat_stats.values())
    tn = sum(cs["tn"] for cs in cat_stats.values())
    fn = sum(cs["fn"] for cs in cat_stats.values())

    prec_g = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_g  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_g   = 2 * prec_g * rec_g / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0.0

    prec_u = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_u  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_u   = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) > 0 else 0.0

    macro_f1 = (f1_g + f1_u) / 2.0

    print("\n" + "=" * 80)
    print("  OVERALL METRICS")
    print("=" * 80)
    print(f"  Total Test Cases:              {n_total}")
    print(f"  Overall Accuracy:              {n_correct} / {n_total} ({n_correct/n_total*100:.1f}%)")
    print(f"  True Positives (Grounded):     {tp}")
    print(f"  False Positives (Halluc. Missed): {fp}")
    print(f"  True Negatives (Halluc. Caught):  {tn}")
    print(f"  False Negatives (Valid Rejected): {fn}")
    print()
    print(f"  Grounded (Class 1) Precision:  {prec_g*100:.1f}%")
    print(f"  Grounded (Class 1) Recall:     {rec_g*100:.1f}%")
    print(f"  Grounded (Class 1) F1:         {f1_g*100:.1f}%")
    print()
    print(f"  Ungrounded (Class 0) Precision:{prec_u*100:.1f}%")
    print(f"  Ungrounded Recall (Catch Rate):{rec_u*100:.1f}%")
    print(f"  Ungrounded (Class 0) F1:       {f1_u*100:.1f}%")
    print()
    print(f"  Macro-F1:                      {macro_f1*100:.1f}%")

    print("\n" + "=" * 80)
    print("  PER-CATEGORY RESULTS")
    print("=" * 80)
    print(f"{'Category':<32} {'Cases':<8} {'Correct':<10} {'Accuracy':<10} {'F1':<10}")
    print("-" * 80)
    for cat, cs in sorted(cat_stats.items()):
        acc = cs["correct"] / cs["total"] if cs["total"] > 0 else 0.0
        # category F1
        c_tp = cs["tp"] + cs["tn"]
        c_fp = cs["fp"]
        c_fn = cs["fn"]
        c_f1 = 2 * c_tp / (2 * c_tp + c_fp + c_fn) if (2 * c_tp + c_fp + c_fn) > 0 else 0.0
        print(f"{cat:<32} {cs['total']:<8} {cs['correct']:<10} {acc*100:>6.1f}%    {c_f1*100:>6.1f}%")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print("\n  FAILURES:")
        for r in failures:
            print(f"    [{r['id']}] {r['category']} - Expected: {r['expected']}, Got: {r['predicted']}, p_true={r['p_true']:.4f}")
            print(f"      {r['desc']}")
    else:
        print("\n  ALL REGRESSION TEST CASES PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    run_regression()
