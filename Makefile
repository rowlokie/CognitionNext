## Makefile for Decide - CPU-only typed decision engine

# Serve the API
serve:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning

# Train production model
train:
	python scripts/resave_baseline.py

# Phase 0: Data audit
audit:
	python scripts/phase0_audit.py

# Evaluate all candidate models
evaluate:
	python scripts/evaluate_candidates.py

# Phase 3: Calibration and threshold tuning
calibrate:
	python scripts/phase3_calibration.py

# Run test suites (server must be running)
test:
	python tests/test_schema.py
	python tests/test_inference.py
	python tests/test_errors.py

# Run self-check (server must be running)
selfcheck:
	python selfcheck.py
	python selfcheck.py --file demo_examples.json

# Benchmark latency (server must be running)
bench:
	python scripts/bench.py

# Install dependencies
install:
	pip install -r requirements.txt

# All metrics from scratch (requires baseline model)
metrics: calibrate bench

.PHONY: serve train audit evaluate calibrate test selfcheck bench install metrics
