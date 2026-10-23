# Latency & Throughput Benchmark Report

Measured via `scripts/bench.py` on 500 warm sequential requests per task directly over HTTP loopback (http://127.0.0.1:8000).

## Summary Table

| Decision Task | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Errors | Budget Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`route`** | 500 | 6.97 | **10.53** | 13.87 | 7.45 | 0 | PASS (<=25ms full marks) |
| **`approve`** | 500 | 6.11 | **7.52** | 9.72 | 6.31 | 0 | PASS (<=25ms full marks) |
| **`grounded`** | 500 | 6.47 | **9.43** | 11.53 | 6.86 | 0 | PASS (<=25ms full marks) |

## Benchmark Configuration
- **Server**: `http://127.0.0.1:8000` (FastAPI + Uvicorn)
- **Protocol**: Sequential HTTP POST `/v1/decide` requests with keep-alive
- **Samples per task**: 500 (1,500 total requests evaluated)
- **Warmup**: 30 pre-benchmark requests

## Artifact Size Audit
- **Artifacts Directory**: `C:\Users\Piyush\Downloads\CognitionNext\artifacts`
- **Total Size**: **16.27 MB**
- **Hard Limit**: 500 MB (Used: 3.25%)
- **Status**: **PASS (Well within 500MB cap)**
