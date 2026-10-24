# Latency & Throughput Benchmark Report

Measured via `scripts/bench.py` on 500 warm sequential requests per task directly over HTTP loopback (http://127.0.0.1:8000).

## Summary Table

| Decision Task | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Errors | Budget Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`route`** | 500 | 11.57 | **16.38** | 18.12 | 11.71 | 0 | PASS (<=25ms full marks) |
| **`approve`** | 500 | 7.67 | **13.50** | 15.62 | 7.21 | 0 | PASS (<=25ms full marks) |
| **`grounded`** | 500 | 11.20 | **15.90** | 18.07 | 11.40 | 0 | PASS (<=25ms full marks) |

## Benchmark Configuration
- **Server**: `http://127.0.0.1:8000` (FastAPI + Uvicorn)
- **Protocol**: Sequential HTTP POST `/v1/decide` requests with keep-alive
- **Samples per task**: 500 (1,500 total requests evaluated)
- **Warmup**: 30 pre-benchmark requests

## Artifact Size Audit
- **Artifacts Directory**: `C:\Users\Piyush\Downloads\CognitionNext\artifacts`
- **Total Size**: **11.26 MB**
- **Hard Limit**: 500 MB (Used: 2.25%)
- **Status**: **PASS (Well within 500MB cap)**
