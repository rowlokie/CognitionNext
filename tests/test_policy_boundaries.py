#!/usr/bin/env python3
"""
tests/test_policy_boundaries.py
Comprehensive Boundary & Policy Verification Test Suite for Decide V2.

Verifies:
1. Below Limit: amounts below configured tier thresholds are eligible for model evaluation.
2. At Limit: boundary inclusivity (amount == limit is allowed when boundary_inclusive=True).
3. Above Limit: amounts strictly above limit are deterministically rejected with clear reason codes.
4. Higher Amounts: extreme amounts are strictly rejected without model bypass.
5. Other Tiers: alias mapping (e.g. standard -> free, business -> pro) and unknown tier safety.
6. Missing / Invalid Amounts: None, negative, NaN, non-numeric coerced gracefully.
7. Epistemic Ambiguity: incomplete evidence tickets trigger uncertainty-aware abstention.
8. Dynamic Configuration: runtime updates to PRO_AUTO_APPROVAL_LIMIT take immediate effect.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from policy import ApprovalPolicyEngine, PolicyConfig, policy_config
from baseline import BaselineModel
import pickle

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def model():
    with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
        mod = pickle.load(f)
    return mod


@pytest.fixture
def policy_engine():
    # Fresh engine with default configuration
    cfg = PolicyConfig({
        "pro_auto_approval_limit": 100.0,
        "free_auto_approval_limit": 50.0,
        "enterprise_auto_approval_limit": 500.0,
        "boundary_inclusive": True,
        "enforce_tier_limits": True,
        "tier_aliases": {
            "standard": "free",
            "business": "pro",
            "premium": "pro",
            "enterprise": "enterprise"
        }
    })
    return ApprovalPolicyEngine(cfg)


# -----------------------------------------------------------------------------
# 1. Below Limit Tests
# -----------------------------------------------------------------------------
def test_pro_tier_below_limit(policy_engine):
    state = {
        "customer_tier": "pro",
        "amount": 99.99,
        "subject": "Minor billing refund",
        "body": "Please refund the mistaken fee."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is True
    assert override is None


def test_free_tier_below_limit(policy_engine):
    state = {
        "customer_tier": "free",
        "amount": 49.99,
        "subject": "Small adjustment",
        "body": "Adjustment for overbill."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is True
    assert override is None


# -----------------------------------------------------------------------------
# 2. At Limit Boundary Tests
# -----------------------------------------------------------------------------
def test_pro_tier_exact_limit_inclusive(policy_engine):
    policy_engine.config.set("boundary_inclusive", True)
    state = {
        "customer_tier": "pro",
        "amount": 100.00,
        "subject": "Refund at threshold",
        "body": "Exact $100 refund."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is True
    assert override is None


def test_pro_tier_exact_limit_exclusive(policy_engine):
    policy_engine.config.set("boundary_inclusive", False)
    state = {
        "customer_tier": "pro",
        "amount": 100.00,
        "subject": "Refund at threshold",
        "body": "Exact $100 refund with exclusive boundary."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is False
    assert override["reason"] == "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED"
    assert override["value"] is False
    assert override["abstain"] is False


def test_free_tier_exact_limit_inclusive(policy_engine):
    policy_engine.config.set("boundary_inclusive", True)
    state = {
        "customer_tier": "free",
        "amount": 50.00,
        "subject": "Free tier $50 refund",
        "body": "Exact $50 request."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is True
    assert override is None


# -----------------------------------------------------------------------------
# 3. Above Limit Tests
# -----------------------------------------------------------------------------
def test_pro_tier_above_limit(policy_engine):
    state = {
        "customer_tier": "pro",
        "amount": 100.01,
        "subject": "Slightly over limit",
        "body": "Requesting $100.01 refund."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is False
    assert override["value"] is False
    assert override["abstain"] is False
    assert override["reason"] == "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED"


def test_free_tier_above_limit(policy_engine):
    state = {
        "customer_tier": "free",
        "amount": 50.01,
        "subject": "Free tier over limit",
        "body": "Requesting $50.01 refund."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is False
    assert override["value"] is False
    assert override["abstain"] is False
    assert override["reason"] == "POLICY_REJECT_FREE_TIER_LIMIT_EXCEEDED"


# -----------------------------------------------------------------------------
# 4. Higher Amounts & Enterprise Tier Tests
# -----------------------------------------------------------------------------
def test_extreme_amounts_rejected(policy_engine):
    for amt in [150.0, 500.0, 2500.0]:
        state = {
            "customer_tier": "pro",
            "amount": amt,
            "subject": "Large refund request",
            "body": f"Please refund ${amt}."
        }
        eligible, override = policy_engine.evaluate_policy(state)
        assert eligible is False
        assert override["value"] is False
        assert override["reason"] == "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED"


def test_enterprise_tier_within_limit(policy_engine):
    state = {
        "customer_tier": "enterprise",
        "amount": 250.0,
        "subject": "Enterprise SLA refund",
        "body": "Uptime fell below contractual threshold."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is True
    assert override is None


def test_enterprise_tier_exceeding_limit(policy_engine):
    state = {
        "customer_tier": "enterprise",
        "amount": 1200.0,
        "subject": "Massive enterprise claim",
        "body": "High-value claim requiring manual executive review."
    }
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is False
    assert override["reason"] == "POLICY_REJECT_ENTERPRISE_LIMIT_EXCEEDED"


# -----------------------------------------------------------------------------
# 5. Other Tiers and Tier Aliases
# -----------------------------------------------------------------------------
def test_tier_aliases(policy_engine):
    # 'standard' maps to 'free' (limit $50)
    state_std_ok = {"customer_tier": "standard", "amount": 25.0, "subject": "OK", "body": "OK"}
    eligible, _ = policy_engine.evaluate_policy(state_std_ok)
    assert eligible is True

    state_std_over = {"customer_tier": "standard", "amount": 75.0, "subject": "Over", "body": "Over"}
    eligible, override = policy_engine.evaluate_policy(state_std_over)
    assert eligible is False
    assert override["reason"] == "POLICY_REJECT_FREE_TIER_LIMIT_EXCEEDED"

    # 'business' maps to 'pro' (limit $100)
    state_biz_ok = {"customer_tier": "business", "amount": 80.0, "subject": "OK", "body": "OK"}
    eligible, _ = policy_engine.evaluate_policy(state_biz_ok)
    assert eligible is True

    state_biz_over = {"customer_tier": "business", "amount": 120.0, "subject": "Over", "body": "Over"}
    eligible, override = policy_engine.evaluate_policy(state_biz_over)
    assert eligible is False
    assert override["reason"] == "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED"


def test_unknown_tier_handled_safely(policy_engine):
    state = {"customer_tier": "custom_vip", "amount": 25.0, "subject": "Test", "body": "Test"}
    eligible, override = policy_engine.evaluate_policy(state)
    assert eligible is True


# -----------------------------------------------------------------------------
# 6. Missing / Invalid Amounts
# -----------------------------------------------------------------------------
def test_missing_or_invalid_amounts(policy_engine):
    # None amount
    state_none = {"customer_tier": "pro", "amount": None, "subject": "None", "body": "None"}
    eligible, _ = policy_engine.evaluate_policy(state_none)
    assert eligible is True

    # Negative amount
    state_neg = {"customer_tier": "pro", "amount": -50.0, "subject": "Negative", "body": "Negative"}
    eligible, _ = policy_engine.evaluate_policy(state_neg)
    assert eligible is True

    # NaN / String amount
    state_str = {"customer_tier": "pro", "amount": "invalid_val", "subject": "String", "body": "String"}
    eligible, _ = policy_engine.evaluate_policy(state_str)
    assert eligible is True


# -----------------------------------------------------------------------------
# 7. Epistemic Ambiguity / Incomplete Evidence Tests
# -----------------------------------------------------------------------------
def test_epistemic_uncertainty_abstains(policy_engine):
    uncertainty_cases = [
        "Payment discrepancy confirmed but logs are not yet available",
        "Discrepancy reported, but an administrator has not confirmed the ledger",
        "Hardware replacement requested; purchase receipt not yet been verified",
        "User filed claim but photo inconclusive regarding shipping damage",
        "Dispute initiated, case pending review by fraud team"
    ]
    for text in uncertainty_cases:
        state = {
            "customer_tier": "pro",
            "amount": 25.0,  # well below limit
            "subject": "Claim status",
            "body": text
        }
        eligible, override = policy_engine.evaluate_policy(state)
        assert eligible is False
        assert override["abstain"] is True
        assert override["reason"] == "EPISTEMIC_UNCERTAINTY_INCOMPLETE_EVIDENCE"


# -----------------------------------------------------------------------------
# 8. Dynamic Configuration Updates
# -----------------------------------------------------------------------------
def test_dynamic_limit_reconfiguration(policy_engine):
    # Update limit to $150.0
    policy_engine.config.set("pro_auto_approval_limit", 150.0)

    state_125 = {"customer_tier": "pro", "amount": 125.0, "subject": "Test", "body": "Test"}
    eligible, _ = policy_engine.evaluate_policy(state_125)
    assert eligible is True

    state_160 = {"customer_tier": "pro", "amount": 160.0, "subject": "Test", "body": "Test"}
    eligible, override = policy_engine.evaluate_policy(state_160)
    assert eligible is False
    assert override["reason"] == "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED"

    # Reset back to $100.0
    policy_engine.config.set("pro_auto_approval_limit", 100.0)
    eligible, override = policy_engine.evaluate_policy(state_125)
    assert eligible is False


# -----------------------------------------------------------------------------
# 9. End-to-End Predict Integration Test
# -----------------------------------------------------------------------------
def test_model_predict_integration(model):
    # Pro tier over limit -> deterministic rejection with reason
    resp_over = model.predict({
        "customer_tier": "pro",
        "amount": 125.0,
        "subject": "Refund request",
        "body": "Standard refund."
    }, "approve", "req-01")
    assert resp_over["value"] is False
    assert resp_over["abstain"] is False
    assert resp_over["reason"] == "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED"

    # Incomplete evidence -> epistemic abstention with reason
    resp_ambig = model.predict({
        "customer_tier": "pro",
        "amount": 35.0,
        "subject": "Discrepancy issue",
        "body": "Administrator has not confirmed the discrepancy."
    }, "approve", "req-02")
    assert resp_ambig["abstain"] is True
    assert resp_ambig["reason"] == "EPISTEMIC_UNCERTAINTY_INCOMPLETE_EVIDENCE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
