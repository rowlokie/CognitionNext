# Latency & Throughput Benchmark Report

Measured via `scripts/bench.py` on 500 warm sequential requests per task directly over HTTP loopback (http://127.0.0.1:8000).

## Summary Table

| Decision Task | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Errors | Budget Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`route`** | 500 | 7.02 | **8.89** | 10.56 | 7.23 | 0 | PASS (<=25ms full marks) |
| **`approve`** | 500 | 6.41 | **8.25** | 9.68 | 6.63 | 0 | PASS (<=25ms full marks) |
| **`grounded`** | 500 | 6.71 | **9.60** | 11.39 | 7.12 | 0 | PASS (<=25ms full marks) |

## Benchmark Configuration
- **Server**: `http://127.0.0.1:8000` (FastAPI + Uvicorn)
- **Protocol**: Sequential HTTP POST `/v1/decide` requests with keep-alive
- **Samples per task**: 500 (1,500 total requests evaluated)
- **Warmup**: 30 pre-benchmark requests

## Artifact Size Audit
- **Artifacts Directory**: `C:\Users\Piyush\Downloads\CognitionNext\artifacts`
- **Total Size**: **5.32 MB**
- **Hard Limit**: 500 MB (Used: 1.06%)
- **Status**: **PASS (Well within 500MB cap)**
