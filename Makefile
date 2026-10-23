## Makefile for Decide — CPU-only typed decision engine

# Serve the API (Phase 1 baseline or Phase 2 encoder, whichever is in artifacts/)
serve:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning

# Phase 0: Data audit
audit:
	python scripts/phase0_audit.py

# Phase 1: Train baseline
train-baseline:
	python scripts/resave_baseline.py

# Phase 2: Train encoder model (requires torch, transformers)
train-encoder:
	python src/encoder_model.py

# Phase 3: Calibration and threshold tuning
calibrate:
	python scripts/phase3_calibration.py

# Run schema tests (server must be running)
test:
	python tests/test_schema.py

# Run self-check (server must be running)
selfcheck:
	python selfcheck.py

# Benchmark latency (server must be running)
bench:
	python scripts/bench.py

# Install dependencies
install:
	pip install -r requirements.txt

# Install encoder dependencies (optional, for Phase 2)
install-encoder:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install sentence-transformers transformers

# All metrics from scratch (requires baseline model)
metrics: calibrate bench

.PHONY: serve audit train-baseline train-encoder calibrate test selfcheck bench install install-encoder metrics
