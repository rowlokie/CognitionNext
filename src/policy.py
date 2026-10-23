#!/usr/bin/env python3
"""
src/policy.py
Configurable Policy Engine for Decide.

Separates deterministic business & support policies from statistical ML model evaluation:
1. Dynamic, configurable policy thresholds (e.g. PRO_AUTO_APPROVAL_LIMIT).
2. Explicitly defined boundary behavior (boundary_inclusive: amount <= limit is allowed).
3. Clear reason codes attached to every decision for auditability.
4. Epistemic uncertainty detection for incomplete / unverified evidence.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "policy.json"
ALT_CONFIG_PATH = ROOT / "artifacts" / "policy.json"

DEFAULT_POLICY = {
    "pro_auto_approval_limit": 100.0,
    "free_auto_approval_limit": 50.0,
    "enterprise_auto_approval_limit": 500.0,
    "boundary_inclusive": True,
    "enforce_tier_limits": True,
    "tier_aliases": {
        "standard": "free",
        "starter": "free",
        "basic": "free",
        "pro": "pro",
        "business": "pro",
        "premium": "pro",
        "enterprise": "enterprise"
    },
    "ineligible_clauses": [
        "non-refundable",
        "nonrefundable",
        "already redeemed",
        "already credited"
    ]
}

UNCERTAINTY_PATTERNS = [
    "not yet available",
    "not recorded",
    "has not confirmed",
    "have not confirmed",
    "has not finished",
    "not complete",
    "incomplete",
    "no criteria supplied",
    "no approver decision",
    "terms are unavailable",
    "photo inconclusive",
    "no photo",
    "inconclusive",
    "logs are not yet available",
    "not yet been verified",
    "pending review"
]


class PolicyConfig:
    """Manages policy settings with hierarchical loading: env vars > config file > defaults."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = dict(DEFAULT_POLICY)
        if config_dict:
            self._config.update(config_dict)
        else:
            self.reload()

    def reload(self):
        loaded = {}
        for path in (CONFIG_PATH, ALT_CONFIG_PATH):
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    break
                except Exception as e:
                    print(f"[PolicyConfig] Warning reading {path}: {e}")
        
        self._config = dict(DEFAULT_POLICY)
        self._config.update(loaded)

        # Environment variable overrides
        if "PRO_AUTO_APPROVAL_LIMIT" in os.environ:
            try:
                self._config["pro_auto_approval_limit"] = float(os.environ["PRO_AUTO_APPROVAL_LIMIT"])
            except ValueError:
                pass

        if "FREE_AUTO_APPROVAL_LIMIT" in os.environ:
            try:
                self._config["free_auto_approval_limit"] = float(os.environ["FREE_AUTO_APPROVAL_LIMIT"])
            except ValueError:
                pass

        if "DECIDE_BOUNDARY_INCLUSIVE" in os.environ:
            self._config["boundary_inclusive"] = os.environ["DECIDE_BOUNDARY_INCLUSIVE"].lower() in ("1", "true", "yes")

        if "ENFORCE_TIER_LIMITS" in os.environ:
            self._config["enforce_tier_limits"] = os.environ["ENFORCE_TIER_LIMITS"].lower() in ("1", "true", "yes")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value

    @property
    def pro_auto_approval_limit(self) -> float:
        return float(self._config.get("pro_auto_approval_limit", 100.0))

    @property
    def free_auto_approval_limit(self) -> float:
        return float(self._config.get("free_auto_approval_limit", 50.0))

    @property
    def enterprise_auto_approval_limit(self) -> Optional[float]:
        val = self._config.get("enterprise_auto_approval_limit")
        return float(val) if val is not None else None

    @property
    def boundary_inclusive(self) -> bool:
        return bool(self._config.get("boundary_inclusive", True))

    @property
    def enforce_tier_limits(self) -> bool:
        return bool(self._config.get("enforce_tier_limits", True))

    @property
    def tier_aliases(self) -> Dict[str, str]:
        return dict(self._config.get("tier_aliases", DEFAULT_POLICY["tier_aliases"]))

    @property
    def ineligible_clauses(self) -> list:
        return list(self._config.get("ineligible_clauses", DEFAULT_POLICY["ineligible_clauses"]))


# Global policy config instance
policy_config = PolicyConfig()


class ApprovalPolicyEngine:
    """
    Deterministic Approval Policy Engine.
    Evaluates policy constraints BEFORE model evaluation.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or policy_config

    def normalize_tier(self, raw_tier: Optional[str]) -> str:
        if not raw_tier:
            return "free"
        norm = str(raw_tier).strip().lower()
        aliases = self.config.tier_aliases
        return aliases.get(norm, norm)

    def normalize_amount(self, raw_amount: Any) -> float:
        if raw_amount is None:
            return 0.0
        try:
            val = float(raw_amount)
            if val != val:  # NaN
                return 0.0
            return max(0.0, val)
        except (TypeError, ValueError):
            return 0.0

    def check_epistemic_uncertainty(self, state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check for missing or unverified evidence."""
        subject = str(state.get("subject") or "").lower()
        body = str(state.get("body") or "").lower()
        full_text = f"{subject} {body}"

        for pat in UNCERTAINTY_PATTERNS:
            if pat in full_text:
                return True, f"EPISTEMIC_UNCERTAINTY: {pat}"
        return False, None

    def evaluate_policy(self, state: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Evaluate deterministic policies for an approval request.
        
        Returns:
            (eligible, override_result)
            - If eligible is False, override_result is the complete policy decision.
            - If eligible is True, request should proceed to statistical ML model evaluation.
        """
        # 1. Epistemic uncertainty check: incomplete evidence warrants abstention
        has_uncertainty, reason = self.check_epistemic_uncertainty(state)
        if has_uncertainty:
            return False, {
                "value": False,
                "p_true": 0.50,
                "abstain": True,
                "reason": "EPISTEMIC_UNCERTAINTY_INCOMPLETE_EVIDENCE",
                "policy_note": reason
            }

        # 2. Extract sanitized tier and amount
        tier = self.normalize_tier(state.get("customer_tier"))
        amount = self.normalize_amount(state.get("amount"))
        inclusive = self.config.boundary_inclusive

        # 3. Deterministic tier limits check
        if self.config.enforce_tier_limits:
            # Pro Tier limit check
            if tier == "pro":
                limit = self.config.pro_auto_approval_limit
                exceeded = (amount > limit) if inclusive else (amount >= limit)
                if exceeded:
                    return False, {
                        "value": False,
                        "p_true": 0.05,
                        "abstain": False,
                        "reason": "POLICY_REJECT_PRO_TIER_LIMIT_EXCEEDED",
                        "policy_note": f"Amount ${amount:.2f} exceeds Pro tier limit of ${limit:.2f}"
                    }

            # Free Tier limit check
            elif tier == "free":
                limit = self.config.free_auto_approval_limit
                exceeded = (amount > limit) if inclusive else (amount >= limit)
                if exceeded:
                    return False, {
                        "value": False,
                        "p_true": 0.05,
                        "abstain": False,
                        "reason": "POLICY_REJECT_FREE_TIER_LIMIT_EXCEEDED",
                        "policy_note": f"Amount ${amount:.2f} exceeds Free tier limit of ${limit:.2f}"
                    }

            # Enterprise Tier limit check (if configured)
            elif tier == "enterprise":
                limit = self.config.enterprise_auto_approval_limit
                if limit is not None:
                    exceeded = (amount > limit) if inclusive else (amount >= limit)
                    if exceeded:
                        return False, {
                            "value": False,
                            "p_true": 0.05,
                            "abstain": False,
                            "reason": "POLICY_REJECT_ENTERPRISE_LIMIT_EXCEEDED",
                            "policy_note": f"Amount ${amount:.2f} exceeds Enterprise tier limit of ${limit:.2f}"
                        }

        # 4. Ineligible clauses check (e.g. non-refundable contracts, already redeemed)
        subject = str(state.get("subject") or "").lower()
        body = str(state.get("body") or "").lower()
        full_text = f"{subject} {body}"
        for clause in self.config.ineligible_clauses:
            if clause in full_text:
                return False, {
                    "value": False,
                    "p_true": 0.05,
                    "abstain": False,
                    "reason": "POLICY_REJECT_INELIGIBLE_CLAUSE",
                    "policy_note": f"Request contains ineligible policy clause: {clause!r}"
                }

        # 5. Passed all deterministic policy checks: request is eligible for model evaluation
        return True, None
