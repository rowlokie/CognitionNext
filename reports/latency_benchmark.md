# Latency & Throughput Benchmark Report

Measured via `scripts/bench.py` on 500 warm sequential requests per task directly over HTTP loopback (http://127.0.0.1:8000).

## Summary Table

| Decision Task | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Errors | Budget Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`route`** | 500 | 14.53 | **20.94** | 24.28 | 15.39 | 0 | PASS (<=25ms full marks) |
| **`approve`** | 500 | 9.71 | **14.83** | 17.69 | 8.44 | 0 | PASS (<=25ms full marks) |
| **`grounded`** | 500 | 14.58 | **21.41** | 25.85 | 15.33 | 0 | PASS (<=25ms full marks) |

## Benchmark Configuration
- **Server**: `http://127.0.0.1:8000` (FastAPI + Uvicorn)
- **Protocol**: Sequential HTTP POST `/v1/decide` requests with keep-alive
- **Samples per task**: 500 (1,500 total requests evaluated)
- **Warmup**: 30 pre-benchmark requests

## Artifact Size Audit
- **Artifacts Directory**: `E:\CongnizationNext\artifacts`
- **Total Size**: **26.64 MB**
- **Hard Limit**: 500 MB (Used: 5.33%)
- **Status**: **PASS (Well within 500MB cap)**
