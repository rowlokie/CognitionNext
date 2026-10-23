# Decide — CPU-Only Typed Decision Engine

A single-endpoint ML service answering three typed decisions about customer-support tickets.
No network calls, no LLMs, no GPU required.

## What It Does

`POST /v1/decide` takes a customer support ticket and a decision type, returns a fully typed response:

| Decision | Type | Output |
|----------|------|--------|
| `route` | choice | Which department: billing/technical/account/shipping/general |
| `approve` | binary | Should we approve this request? (yes/no + calibrated probability) |
| `grounded` | binary | Is the draft reply factually grounded in the KB article? |

## Request/Response Examples

### Route
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-001",
    "decision": "route",
    "state": {
      "subject": "Invoice discrepancy",
      "body": "I was charged twice this month",
      "customer_tier": "pro",
      "amount": 99.0,
      "draft_reply": "",
      "kb_article": ""
    }
  }'
```
Response:
```json
{
  "request_id": "req-001",
  "decision": "route",
  "type": "choice",
  "value": "billing",
  "probs": {"billing": 0.821, "technical": 0.042, "account": 0.091, "shipping": 0.008, "general": 0.038},
  "confidence": 0.821,
  "abstain": false,
  "latency_ms": 4.2
}
```

### Approve
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-002",
    "decision": "approve",
    "state": {
      "subject": "Refund request",
      "body": "Please refund my $15 charge",
      "customer_tier": "enterprise",
      "amount": 15.0,
      "draft_reply": "",
      "kb_article": ""
    }
  }'
```
Response:
```json
{
  "request_id": "req-002",
  "decision": "approve",
  "type": "binary",
  "value": true,
  "p_true": 0.892,
  "abstain": false,
  "latency_ms": 3.8
}
```

### Grounded
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-003",
    "decision": "grounded",
    "state": {
      "subject": "Warranty",
      "body": "Is my product under warranty?",
      "customer_tier": "free",
      "amount": 0.0,
      "draft_reply": "All physical goods carry six months warranty from delivery.",
      "kb_article": "The warranty covers all physical goods for six months from delivery."
    }
  }'
```
Response:
```json
{
  "request_id": "req-003",
  "decision": "grounded",
  "type": "binary",
  "value": true,
  "p_true": 0.784,
  "abstain": false,
  "latency_ms": 5.1
}
```

## Where the Shared Backbone Lives

**Single backbone with three task heads** (not three separate models):

```
src/baseline.py — BaselineModel (Phase 1 Baseline & Production Model)
├── shared_tfidf_word / shared_tfidf_char (ONE shared representation backbone across all tasks)
│   ├── word n-grams (1-2), 50K max features, sublinear TF
│   └── char n-grams (3-5), 30K max features, sublinear TF
├── structured features (tier one-hot, log1p amount, amount bands, length & overlap features)
│
└── predict(state, decision) → full response dict
    ├── "route"    → route_model (Calibrated multiclass LR over 5 queues)
    ├── "approve"  → approve_model (Calibrated binary LR + tuned 0.700 cost threshold)
    └── "grounded" → grounded_model (Calibrated binary LR + lexical overlap features)

src/encoder_model.py — SharedEncoderModel (Phase 2 Neural Alternative)
├── encoder: all-MiniLM-L6-v2 (ONE shared learned transformer backbone)
├── route_head: Linear(384+14 → 256 → 5)
├── approve_head: Linear(384+14 → 128 → 1)
└── grounded_head: Linear(384+14 → 128 → 1)
```

The `predict(state, decision)` interface is identical — `app/main.py` uses whichever model is present in `artifacts/`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Audit data (Phase 0)
make audit

# Train baseline (Phase 1)
make train-baseline

# Calibrate thresholds (Phase 3)
make calibrate

# Start server (always uses best available model)
make serve
# or directly:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Run schema tests (server must be running)
make test

# Run self-check
make selfcheck

# Benchmark latency
make bench
```

## Phase 2 Encoder (Optional — CPU, requires ~500MB download)

```bash
# Install encoder deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers

# Train shared encoder
make train-encoder

# Re-calibrate
make calibrate
```

## CPU-Only and No-Network Statement

> **This service makes zero network calls at runtime or training time.**
> All inference is local. No LLM APIs, no hosted model APIs, no external services.
> Runs on standard 4-core CPU. GPU is not required.

## Pretrained Components and Licenses

| Component | Source | License |
|-----------|--------|---------|
| `all-MiniLM-L6-v2` (Phase 2) | sentence-transformers/HuggingFace | Apache 2.0 |
| scikit-learn | scikit-learn.org | BSD-3-Clause |
| FastAPI | tiangolo/fastapi | MIT |
| orjson | ijl/orjson | MIT |

## Runtime Versions

| Package | Version |
|---------|---------|
| Python | 3.11.9 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| fastapi | 0.143.2 |
| uvicorn | 0.54.0 |
| orjson | 3.13.0 |
| pydantic | 2.14.1 |

## Artifact Size & Audit

- `artifacts/baseline_model.pkl`: **7.69 MB** (compact multi-task representation backbone)
- `artifacts/thresholds.json`: **<1 KB** (calibrated decision and abstention thresholds)
- `artifacts/policy.json`: **<1 KB** (configurable approval policy thresholds)
- Total production footprint: **~7.7 MB** (only 1.5% of the 500 MB hackathon limit)

## Approval Policy Engine (Decide V2)

Decide V2 introduces a dedicated, configurable **Approval Policy Engine** (`src/policy.py`) that separates deterministic business rules and SLA contracts from statistical ML model estimation:
- **Configurable Limits**: Default Pro auto-approval limit at $100.00 (`config/policy.json`), dynamically overridable via environment variables.
- **Explicit Boundary Behavior**: Boundary inclusive semantics (`amount <= limit` allowed, `amount > limit` strictly rejected).
- **Epistemic Uncertainty Abstention**: Detects incomplete / unverified evidence and triggers deliberate abstention.
- **Strict Public API Schema**: Responses strictly conform to the 7-field binary and 8-field choice specifications with 0 extra keys.

## Project Layout

```
app/          FastAPI application (main.py)
src/          Model code (baseline.py, policy.py, encoder_model.py)
scripts/      Phase scripts (audit, calibration, bench, eval)
tests/        Schema, error, inference, and policy boundary tests
artifacts/    Trained model artifacts (<27 MB total)
config/       Policy configuration (policy.json)
data/         train.jsonl, dev.jsonl
docs/         Playbook and audit documentation
reports/      Benchmark and evaluation reports
results.json  Measured benchmark results
model_card.md Model card documentation
selfcheck.py  Schema validation tool
demo_examples.json  Example requests
Makefile      All single-command entry points
requirements.txt  Pinned dependencies
```
