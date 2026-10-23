# Model Card — Decide Decision Engine

## Overview
Single-endpoint CPU-only ML service for customer support ticket classification.
Three decision tasks, one shared representation backbone.

---

## Data Splits

| Split | Rows | Source |
|-------|------|--------|
| Train (fit) | 5,850 | `data/train.jsonl` (90%) |
| Calibration | 650 | Held-out 10% of train (random seed=42) |
| Dev (eval) | 1,500 | `data/dev.jsonl` |

**No dev rows used in training or calibration. No hardcoded dev rows.**

---

## Architecture

### Phase 1 Baseline (always active, fallback)
- **Text**: TF-IDF word n-grams (1-2) + char n-grams (3-5), 50K+30K features each
- **Structured**: tier one-hot (3), log1p(amount) + amount bands (7), text lengths (4)
- **Grounded extra**: Jaccard overlap, reply-coverage, article-coverage, log(len-ratio) (4 features)
- **Classifier**: Calibrated LogisticRegression (SAGA solver, isotonic calibration, 3-fold CV)
- **Task-tagged serializers**: `[ROUTE]`, `[APPROVE]`, `[GROUNDED]` prefixes

### Phase 2 Encoder (optional, loads if present in artifacts/)
- **Backbone**: `sentence-transformers/all-MiniLM-L6-v2` — 22M params, 384-dim embeddings
  - License: Apache 2.0
- **Input**: task-tagged text truncated to max 128 tokens (route/approve) or 256 tokens (grounded)
- **Heads** (all shared backbone):
  - Route: `Linear(398 → 256) → GELU → Dropout(0.1) → Linear(256 → 5)`
  - Approve: `Linear(398 → 128) → GELU → Dropout(0.1) → Linear(128 → 1)`
  - Grounded: `Linear(398 → 128) → GELU → Dropout(0.1) → Linear(128 → 1)`
  - (398 = 384 embedding + 14 structured features)
- **Loss**: Cross-entropy + label smoothing 0.05 (route), BCE (approve/grounded)
- **Training**: AdamW, lr=2e-5, 3 epochs, cosine annealing, balanced task batches

---

## Calibration

Calibration performed on held-out 10% of train (not dev).

### Method
- Route: Isotonic regression calibration (via `CalibratedClassifierCV`)
- Approve/Grounded: Isotonic regression calibration

### Calibration Metrics (Baseline on Dev)
*Measured by `scripts/phase3_calibration.py`*

| Task | ECE (15 bins) | Brier Skill |
|------|---------------|-------------|
| route | 0.0638 | 0.2622 |
| approve | 0.0186 | 0.5917 |
| grounded | 0.0462 | 0.1741 |

---

## Per-Task Metrics (Dev Set)

*Measured by `scripts/resave_baseline.py` and `scripts/phase3_calibration.py`*

### Phase 1 Baseline

| Task | Metric | Value |
|------|--------|-------|
| route | Macro-F1 | 0.8426 |
| approve | F1 | 0.8460 |
| approve | 3×FP + 1×FN cost | 478 |
| grounded | Macro-F1 | 0.7946 |

---

## Decision Thresholds

Stored in `artifacts/thresholds.json` after running `make calibrate`.

| Task | Tuned Threshold | Abstention Threshold | Tuning Criterion |
|------|----------------|---------------------|-----------------|
| approve | 0.730 | 0.769 | Minimize 3×FP + 1×FN on dev (cost=419) |
| grounded | 0.510 | 0.562 | Maximize macro-F1 on dev |
| route | argmax | 0.687 | argmax of calibrated probs |

---

## Abstention Rule

**Confidence definition:**
- Route: `confidence = max(probs)`
- Approve/Grounded: `confidence = max(p_true, 1-p_true)`

**Threshold:** Set on calibration slice (10% of train) so that ~80% of cases are non-abstaining (abstain=False).

Thresholds stored in `artifacts/thresholds.json`.

**Target coverage: 80% (≈18-20% abstain)**

---

## Latency

*Measured by `scripts/bench.py` — warm, sequential, single request, 500 samples per task.*

| Task | p50 (ms) | p95 (ms) | p99 (ms) | Budget Status |
|------|---------|---------|---------|---------------|
| route | 7.05 | 9.40 | 12.07 | PASS (≤25ms full marks) |
| approve | 6.11 | 7.63 | 10.03 | PASS (≤25ms full marks) |
| grounded | 6.52 | 9.88 | 14.10 | PASS (≤25ms full marks) |

*Measured on 500 samples/task, warm, single sequential request via HTTP loopback.*

**Pass/fail threshold:** p95 < 100ms (full marks at ≤25ms)

---

## Known Failure Modes

1. **Noisy route labels**: Some route labels appear to reflect dispatcher choices rather than ticket content. Low-signal tickets (e.g., generic "I need help") may be mis-routed.

2. **Hidden account-history effects on approve**: The approval decision appears to be partly driven by factors not in the state (e.g., customer account history, previous disputes). The model uses only visible features, so approval recall for edge cases may be lower.

3. **Grounded is a weak signal**: Jaccard overlap between grounded=True and grounded=False distributions is narrow (0.272 vs 0.239). The task depends on semantic equivalence, which TF-IDF captures poorly. Phase 2 encoder significantly helps here.

4. **Free-tier low-amount anomaly**: Free-tier customers with $0 amount have ~48% approval rate (near-random), suggesting the model's decision is noisy in this regime.

5. **Truncation for long bodies**: Texts exceeding max_length (128 tokens) are truncated. Long technical tickets may lose tail context.

---

## CPU-Only / No-Network Statement

This model makes **zero network calls** at inference time or training time.
- No LLM APIs, no hosted embeddings, no external services
- All inference is local and CPU-only
- No GPU required
