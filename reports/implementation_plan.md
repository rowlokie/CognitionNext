# DECIDE Hackathon — Complete Implementation Plan

## 1. Official Constraints & Judge Gates
- **Shared Backbone**: Exactly one shared feature/encoder representation for all three decision tasks (`route`, `approve`, `grounded`). Task-specific heads and calibration are permitted and required.
- **Serving Endpoint**: `POST /v1/decide` accepting `{request_id, decision, state}` and responding with typed schemas:
  - `route`: `type='choice'`, `value` in `[billing, technical, account, shipping, general]`, `probs` (5 classes, sum to 1 ± 0.01), `confidence = probs[value]`, `abstain` (bool), `latency_ms` (numeric).
  - `approve` & `grounded`: `type='binary'`, `value` (bool), `p_true` in `[0, 1]`, `abstain` (bool), `latency_ms` (numeric).
- **Latency Budget**: Single-request sequential warm HTTP latency $p_{95} < 100\text{ ms}$ (full marks at $\le 25\text{ ms}$).
- **Hardware & Offline Rules**: CPU-only, strictly zero network calls during training, serving, or inference. Zero hosted LLMs/APIs.
- **Artifact Size**: Must be under 500 MB total.
- **Asymmetric Cost on `approve`**: Cost metric $= 3 \times \text{FP} + 1 \times \text{FN}$. Optimal threshold theoretically $\approx 0.75$; tuned empirically on validation/dev.
- **Abstention Policy**: Coverage target $\approx 80\%$, error evaluated on top $80\%$ most confident samples.
- **Calibration (Guo et al. 2017)**: Calibrate on held-out 650-row train slice; report ECE (15 bins) and Brier skill score.

---

## 2. Existing Implementation & Baseline Metrics (Reference)
- **Architecture**: Shared TF-IDF word (1-2) + char (3-5) n-grams + structured features (tier, log-amount, lengths, text overlap) with 3 linear heads.
- **Baseline Metrics (Unseen Dev, 1,500 samples)**:
  - `route`: Macro-F1 = 84.26%, Accuracy = 84.53%, ECE = 0.0638
  - `approve`: Accuracy = 86.53%, Precision = 80.09%, Recall = 89.66%, F1 = 84.60%, 3:1 Cost = 419 (at $p \ge 0.730$), ECE = 0.0186
  - `grounded`: Accuracy = 71.53%, Precision = 72.20%, Recall = 88.34%, F1 = 79.46% (Macro-F1 = 66.86%), ECE = 0.0462
  - **Latency**: $p_{50} = 6.1 - 7.1\text{ ms}$, $p_{95} = 7.6 - 9.9\text{ ms}$ (comfortably within the $\le 25\text{ ms}$ full marks bracket).
  - **Artifact Size**: 5.3 MB (well below 500 MB limit).

---

## 3. Discrepancies & Missing Elements Identified
- Need systematic evaluation across all candidates:
  - Candidate A: Reference TF-IDF baseline.
  - Candidate B: Enhanced linear models (feature scaling, regularizations, class weighting, enriched grounding interactions).
  - Candidate C: Neural shared encoder (`all-MiniLM-L6-v2`, available offline in HuggingFace cache) with multi-task heads.
  - Candidate D: Tree-based structured baseline (scikit-learn `HistGradientBoostingClassifier`).
  - Candidate E: Hybrid architecture (Shared text representation + dense structured features).
- Need formal reporting artifacts:
  - `reports/data_audit.md`
  - `reports/model_comparison.csv`
  - `reports/model_comparison.md`
  - `reports/final_model_selection.md`
  - `reports/latency_benchmark.md`
- Need comprehensive test suite covering schema, inference, and error paths (`tests/test_schema.py`, `tests/test_inference.py`, `tests/test_errors.py`).

---

## 4. Experimentation & Candidate Evaluation Plan
1. **Candidate A**: Benchmark and freeze current TF-IDF baseline metrics.
2. **Candidate B**: Systematic grid of n-gram limits, $C$ penalties, class weighting, and sublinear scaling.
3. **Candidate C**: Multi-task MiniLM fine-tuning on CPU (1-2 epochs with batching and cosine decay) to measure F1 gains vs. latency cost.
4. **Candidate D**: Gradient-boosted trees on structured + lexical overlap features.
5. **Candidate E**: Enriched linear multi-task backbone with tuned cross-task representations.

---

## 5. Checklist of Completion Criteria
- [x] Phase 0: Repository, playbooks, docs inspected, git checkpoint branch created.
- [ ] Phase 1: Full data audit executed, line-by-line JSONL validation, leak checks, `reports/data_audit.md` generated.
- [ ] Phase 2: Baseline verified and preserved as fallback.
- [ ] Phase 3: All 5 candidates trained, evaluated, and documented in `reports/model_comparison.csv` and `reports/model_comparison.md`.
- [ ] Phase 4: Calibration (isotonic vs Platt) and thresholding tuned on 650-row held-out split; abstention error evaluated at 80% coverage.
- [ ] Phase 5: Final model selected against all hard constraints and documented in `reports/final_model_selection.md`.
- [ ] Phase 6: Production API endpoint `POST /v1/decide` updated with full error handling and test suite (`test_schema.py`, `test_inference.py`, `test_errors.py`).
- [ ] Phase 7: Full HTTP latency benchmarking over 500 requests per task, results written to `reports/latency_benchmark.md` and `results.json`.
- [ ] Phase 8: Full verification with clean restart and `selfcheck.py`.
- [ ] Phase 9: `README.md`, `model_card.md`, and `Makefile` updated and validated.
