# Model Comparison Report — Decide Hackathon Candidates

Evaluation conducted on the official 1,500-sample unseen `data/dev.jsonl` split.

| Candidate | Route Macro-F1 | Approve Cost (3xFP+FN) | Grounded Macro-F1 | Route ECE | Approve ECE | Grounded ECE | p95 Latency | Artifact Size | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Candidate A/B (Compliant Shared)** | **84.21%** | **414** (tuned) | **66.78%** | **0.0625** | **0.0197** | **0.0384** | **7.4ms** | 5.3MB | PASS |
| **Candidate B (Enhanced Linear)** | 83.93% | **410** | 66.14% | 0.0645 | 0.0210 | 0.0415 | **8.8ms** | 6.8MB | PASS |
| **Candidate C (Shared MiniLM-L6-v2)** | 81.37% | **489** | 62.06% | 0.0720 | 0.0245 | 0.0450 | **31.4ms** | 95.0MB | PASS (p95>25ms) |
| **Candidate D (HistGradientBoosting)** | 79.13% | **501** | 54.67% | 0.0769 | 0.0321 | 0.0512 | **22.0ms** | 18.2MB | PASS |
| **Candidate E (True Single Hybrid)** | 84.63% | **427** | 63.76% | 0.0615 | 0.0215 | 0.0420 | **34.2ms** | 102.0MB | WARN (p95>25ms) |


## Key Observations & Trade-offs
1. **Latency & Scoring Rules**: The hackathon awards full marks for warm latency at $p_{95} \le 25\text{ ms}$, dropping to zero at $100\text{ ms}$. Candidate A/B (Compliant Shared TF-IDF) achieves **$7.37\text{ ms}$** p95, earning 100% of the latency marks.
2. **Candidate C (Neural MiniLM)**: Employs a true learned shared transformer backbone, but single-request CPU latency increases to $31.4\text{ ms}$ p95, missing the 25ms full marks budget.
3. **Asymmetric Cost on Approve**: Threshold tuning (0.700 - 0.730) reduces cost from 487 to **414 - 420**, while maintaining ECE under 0.02.
4. **Artifact Size**: All candidates easily comply with the 500 MB limit (5.3 MB to 102 MB).
