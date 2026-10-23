DECIDE — Hackathon Execution Playbook

12-hour, 2-person Applied ML sprint | CPU-only typed decision engine

Prepared from the uploaded Decide problem statement and the supplied train.jsonl/dev.jsonl files

Operating principle: meet the T+2 and T+5:30 floors first; optimize only after the service is judgeable.

1. Executive strategy

Win by shipping a valid system early, then improving measured decision quality without breaking schema or latency. The non-negotiables are: one shared model/backbone for route, approve and grounded; CPU-only offline inference; calibrated probabilities; deliberate abstention; p95 under 100 ms per decision type; artifacts under 500 MB.

Recommended path: start with one shared TF-IDF feature pipeline over all text fields plus structured tier/amount features, and a single multi-task model implementation with task-specific output heads. This gives a strong, fast baseline. If time allows, compare a compact pretrained encoder fine-tuned with a shared backbone and three heads, then export/quantize to ONNX. Keep the classical model as a fallback. Do not deploy three independent models behind a router.

Important distinction: 'one shared model' does not mean one identical label head. Route is 5-way classification; approve is asymmetric-cost binary classification; grounded is evidence/entailment-style binary classification. Share the representation and use task-specific heads/losses/calibration.

2. Hard requirements and judge gates

Endpoint: POST /v1/decide. Request: request_id, decision in {route, approve, grounded}, and state fields subject, body, customer_tier, amount, draft_reply, kb_article. Return every required field exactly as specified in the problem statement.

Route response: type='choice'; value is one of billing, technical, account, shipping, general; probs contains all five queues and sums to 1 within 0.01; confidence equals the probability of the selected value; abstain is boolean; latency_ms is numeric.

Approve/grounded response: type='binary'; value is boolean; p_true is calibrated probability of True in [0,1]; abstain is boolean; latency_ms is numeric.

Pass/fail: baseline scored by T+2:00; live schema-valid endpoint by T+5:30; p95 below 100 ms per type; valid every field; model_card.md; public GitHub repo; results.json; README with clean-machine single-command startup; pretrained component/license disclosure; live 3-minute demo. Final submission at T+12:00 is hard deadline.

Serving must make no network calls and require no GPU. Never use a hosted LLM/embedding API in inference. Don't hard-code dev examples. Keep artifacts below 500 MB. Push commits every 30 minutes; preserve code/model artifacts before Lightning Studio restarts.

3. What the provided JSONL actually contains

Measured from the supplied files: train.jsonl has 6,500 records (about 4.82 MB); dev.jsonl has 1,500 records (about 1.11 MB). Each line is a JSON object with request_id, state, labels. All six state fields are present in every inspected record: subject, body, customer_tier, amount, draft_reply, kb_article. Labels contains route, approve, grounded.

Train route labels: billing 1,829 (28.1%), technical 1,541 (23.7%), account 1,324 (20.4%), shipping 998 (15.4%), general 808 (12.4%). Dev route: billing 395 (26.3%), technical 354 (23.6%), account 298 (19.9%), shipping 233 (15.5%), general 220 (14.7%). Use macro-F1, not just accuracy, because class balance is uneven.

Train approve labels: False 4,047 (62.3%), True 2,453 (37.7%). Dev: False 881 (58.7%), True 619 (41.3%). A naive majority predictor is not a meaningful solution. Approval errors are asymmetric: false auto-approval costs 3x a false hold, so choose a decision threshold using the specified cost function and tune it on dev.

Train grounded labels: True 3,887 (59.8%), False 2,613 (40.2%). Dev: True 935 (62.3%), False 565 (37.7%). Grounded must compare draft_reply with kb_article, not just ticket subject/body. Preserve both texts in model input and try an explicit pair format or similarity/overlap features.

Data-use policy: train on train.jsonl; use dev.jsonl for model selection, threshold/calibration tuning, and reported dev metrics only. Never memorize individual dev rows. Do not use hidden data. Keep request_id out of model features. Parse JSONL line-by-line; validate missing/null types and report malformed rows rather than silently shifting labels.

4. Feature and model plan

Fast baseline (first): build a deterministic text serializer with task marker plus fields. For route and approve, include subject + body + customer_tier + amount; include draft_reply + kb_article as useful context only if validation says they help. For grounded, explicitly serialize draft_reply and kb_article together. Add structured features: customer_tier one-hot, log1p(amount), amount bands, text lengths, word/character n-grams.

Baseline candidate A: shared word/character TF-IDF vectorizer and one shared multi-task estimator architecture. A practical implementation is a common sparse representation plus task-specific linear heads (LogisticRegression / LinearSVC with calibrated probabilities). Document the shared vectorizer/backbone and the three heads clearly in README. If strict interpretation requires a single jointly trained model object, implement a shared trunk with three output heads, or use a compact shared encoder with three heads.

Candidate B (if compute/setup allows): compact open-weight encoder, fine-tuned on a task-tagged input such as '[TASK=ROUTE] ...', '[TASK=APPROVE] ...', '[TASK=GROUNDED] draft: ... article: ...', with a shared encoder and three heads. Keep inference CPU-friendly; quantize/export to ONNX only after checking probabilities and output parity. Never assume a transformer will be faster than sparse linear inference.

Grounding-specific improvement: include pairwise lexical overlap, token coverage of claims in the article, and optionally sentence-level maximum similarity as additional features. These should be features in the same shared model pipeline, not a separate deployed model. Beware paraphrases and contradictions; overlap alone is insufficient.

Approval-policy discovery: inspect label rates by customer_tier, amount bins, key phrases, and combinations. Fit a transparent baseline and inspect coefficients/feature importance. Some labels depend on unobserved account history, per the brief, so represent uncertainty rather than trying to force perfect accuracy. Do not hard-code a guessed policy.

Data hygiene: stratify train/dev comparisons by label; check duplicate/near-duplicate text leakage; inspect confusion matrices and per-class F1; preserve dev as a fixed evaluation set. Use class weighting only if dev macro-F1 improves; it can hurt calibration.

5. Calibration, thresholds and abstention

Calibrate probabilities on a held-out slice from train, or use cross-validation to avoid fitting calibration on the same examples used to fit the base estimator. Then evaluate once on dev. For linear classifiers, use sigmoid/Platt calibration or isotonic calibration if enough calibration data; choose by dev Brier score/ECE, not accuracy alone. Temperature scaling is a natural option for neural logits.

Report Expected Calibration Error (ECE) and Brier score/skill where possible. Use reliability bins; compare predicted confidence to empirical accuracy. Do not output 0.99 by default. Route probs must sum to 1 and confidence must be max(probs). For binary decisions, p_true must remain the calibrated probability of True.

Approval threshold: because wrongly approving costs 3 and wrongly holding costs 1, a simple cost-minimizing rule under calibrated probabilities predicts approve only when 3*(1-p_true) < 1*p_true, i.e. p_true > 0.75. Treat this as the theoretical starting point, not an unquestionable final threshold: validate the exact scoring definition and choose/tune on dev using the hackathon's cost-weighted metric.

Abstention: choose a confidence threshold per task using dev. The brief's abstention metric evaluates the most confident 80% regardless of the flag; the flag removes cases from quality scoring, which is coverage-scaled against an 80% target. Therefore do not set abstain flags randomly or use a fixed 0.5 threshold blindly. Rank by confidence (route max probability; binary max(p_true, 1-p_true)), estimate errors at 80% coverage, then select an operational threshold that preserves useful coverage. Always explain that ambiguous cases go to a human.

Record thresholds, calibration method, calibration split, ECE/Brier, coverage, selective error, and the exact cost rule in model_card.md.

6. Two-person division of work

Member A — Modeling/data owner: inspect labels and data; build shared feature/model pipeline; establish baseline; tune route macro-F1, approval cost, grounding macro-F1; calibrate probabilities; produce saved model and training/evaluation scripts.

Member B — API/reliability owner: create FastAPI endpoint and strict request/response validation; implement shared model loader and per-task output formatting; write schema tests/selfcheck; benchmark warm single-request p50/p95; build results.json; README, model_card, demo examples; manage GitHub pushes and runtime.

Pair checkpoints: first 15 minutes agree schema and repo layout; at every milestone both pull/run the latest integrated code. Member B should create a mock/fallback model immediately so endpoint work never waits for model training. Member A must provide a stable predict(state, decision) interface early.

7. Timeboxed plan aligned to 2:00 AM, 5:30 AM and final submission

Assuming the hackathon kickoff is 12 hours before the final deadline and the stated milestones are literal local times: work backward from 2:00 AM as the T+2 baseline floor, 5:30 AM as the T+5:30 live endpoint floor, and final at T+12:00. If kickoff time differs, preserve relative T+ deadlines exactly; they are authoritative.

NOW to 2:00 AM — Phase 0/1 (baseline must be submitted and scored): 0:00–0:15 read schema, create repo and assign roles; 0:15–0:45 parse JSONL, validate labels and build text serializer; 0:45–1:20 fit TF-IDF + fast classifiers; 1:20–1:40 evaluate on dev and save artifacts; 1:40–1:55 submit/push baseline and record dev metrics; 1:55–2:00 verify submission is visible. Do not wait for a neural model.

2:00–5:30 AM — Phase 2 (live endpoint floor): integrate model loader; implement POST /v1/decide; implement exact schemas for all three decisions; add input validation and error handling; warm up and run one request per decision; test probabilities sum to 1; make startup command work from a clean environment; publish and push. By 5:15 AM, freeze features and spend the final 15 minutes on schema and endpoint smoke tests.

5:30–8:00 AM — Phase 3 (quality): improve one task at a time using dev metrics; likely best ROI is grounding pair features and approval policy threshold. Compare candidate changes against baseline and keep only measurable gains. Inspect confusion matrices, label skew, calibration and errors.

8:00–9:30 AM — Phase 4 (calibration/latency): calibrate each task; tune abstention on dev; benchmark warm single-request p50/p95 by decision type (no batching); if p95 is above target, simplify text preprocessing, cap pathological input length sensibly, reduce thread oversubscription, and consider ONNX/quantization only if it is safe.

9:30–10:30 AM — Phase 5 (submission artifacts): generate results.json with per-task p50/p95, dev metrics, calibration metrics and coverage; write one-page model_card.md; README with architecture/shared-backbone explanation, license disclosure, environment, and one startup command; run the provided selfcheck.py if available and own tests.

10:30–11:15 AM — Phase 6 (demo rehearsal): rehearse exactly 3 minutes: 30 sec problem/architecture; 45 sec live route decision with all queue probabilities; 45 sec grounded example that should abstain; 30 sec approval trade-off and 0.75 starting threshold; 30 sec measured latency/calibration and limitations. Ensure the demo uses real requests, not hard-coded outputs.

11:15–12:00 — freeze and submit: clean clone/start test; verify public repo visibility; push all commits and model artifacts; check files, host/port, endpoint reachability, README single command, model card, results, license citations; provide host and port as required; submit before T+12:00. No late architecture changes.

8. Submission repo and exact final checklist

Suggested layout: app/main.py; app/inference.py; src/data.py; src/train.py; src/evaluate.py; src/calibrate.py; tests/test_schema.py; artifacts/shared_model.joblib (or compact model/ONNX files); results.json; model_card.md; README.md; requirements.txt; demo_examples.json; .gitignore.

README must include: what Decide does; request/response examples; architecture diagram or clear text showing shared representation and task-specific heads; training command; evaluation command; single service startup command; CPU-only/no-network inference statement; exact Python/runtime versions; every pretrained model's name and license; artifact-size note.

results.json must include measured p50/p95 latency by route/approve/grounded after warm-up with single requests, plus public dev metrics and calibration results. Never invent benchmark numbers: measure them on the actual runtime and state machine details.

model_card.md (required for ranking): model summary; data splits; shared architecture; feature/task handling; training/calibration process; per-task measured quality; ECE/Brier; abstention coverage/rules; latency; known failure modes including inconsistent route labels and hidden account-history effects.

Before submission: (1) git status clean; (2) git push successful; (3) public repo opens in incognito; (4) clean environment startup works; (5) every decision returns required fields; (6) route probs all five labels and sum to 1; (7) binary p_true valid; (8) abstain boolean; (9) p95 <100ms; (10) artifacts <500MB; (11) no network call in serving path; (12) model card/results/README present; (13) host/port shared; (14) demo fits 3 minutes.

9. Research-paper findings that matter: Guo et al. (2017)

The problem statement cites Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger, 'On Calibration of Modern Neural Networks' (ICML 2017). The key practical finding is that modern neural networks can be accurate yet overconfident; raw softmax probabilities should not automatically be treated as trustworthy confidence.

Temperature scaling is the paper's central practical recommendation: divide logits by one learned scalar temperature before softmax. It is simple, uses a held-out validation set, and often improves calibration while leaving the predicted class unchanged. This is especially relevant when the neural model's accuracy is good but ECE is poor.

Implication for Decide: calibrate on held-out data and report both calibration and task quality. For neural models, temperature scaling is a low-cost first choice. For linear/scikit-learn models, use a calibration method appropriate to the estimator (sigmoid/Platt or isotonic) and select it on held-out validation data. Do not calibrate on dev if you also report dev as untouched final evaluation; reserve a calibration split from train.

Do not overclaim: calibration cannot fix wrong labels, missing account history, poor ranking, or grounding contradictions. It improves the meaning of confidence, not necessarily classification accuracy. Also use Brier score/skill alongside ECE because binning choices can make ECE look better or worse.

10. Doubts to ask the company senior/mentor

1. Is the final deadline exactly T+12:00 from kickoff, and are 2:00 AM and 5:30 AM the official T+2 and T+5:30 timestamps for this cohort?

2. Can you share the exact selfcheck.py and expected schema edge cases (unknown decision, missing fields, null values, extra fields, malformed JSON)?

3. Does 'one model' explicitly permit one shared TF-IDF representation plus three task-specific heads, or do judges require a single jointly trained multi-task model object? The brief says the backbone must be shared; we want to satisfy its intended interpretation.

4. What precisely is the approval cost-weighted error formula and normalization used by the scorer? Is the theoretical p_true > 0.75 threshold the correct decision rule for that metric?

5. How is abstention scored when the flag is true: is there a minimum coverage floor beyond the stated 80%-coverage ranking? Are thresholds task-specific?

6. Are latency measurements taken on the same 4-core/16-GB Lightning CPU shape, and does latency include JSON parsing, preprocessing, and response serialization?

7. Are there constraints on model artifact format or dependency size beyond the 500 MB model-artifact cap? Is ONNX Runtime allowed in the judging environment?

8. What exact host/port handoff format is required at submission, and how should the live service remain available during automated judging?

9. Is there a reference implementation for results.json fields or a sample successful submission format?

10. Can mentors clarify whether the provided train/dev counts are expected (6,500/1,500) and whether any data schema variations are intentional?

11. Risk register and winning priorities

Highest risks: missing early floors; schema mismatch; training three independent models; overfitting dev; uncalibrated confidence; approval threshold at 0.5; grounding model ignoring article/reply relationship; p95 measured incorrectly with batching; artifacts too large; unpushed work lost during Studio restart.

Prioritize in this order: (1) baseline submitted by T+2; (2) valid endpoint by T+5:30; (3) exact schema and CPU-only; (4) measured latency under 100ms; (5) calibration + sensible abstention; (6) task quality improvements on dev; (7) polish docs/demo. A slightly less accurate model that starts reliably and reports honest probabilities beats an impressive notebook that cannot be judged.

Do not claim guaranteed winning. The winning approach is disciplined: protect pass/fail floors, run controlled experiments, keep the best measured candidate, and explain trade-offs with numbers.

Appendix A. Minimal response logic

route: probs = calibrated softmax/probabilities over exactly five queues; value = argmax(probs); confidence = probs[value]; abstain = confidence below chosen route threshold.

approve: p_true = calibrated probability of approve; value = (p_true > tuned_cost_threshold), initially test 0.75; confidence_for_abstention = max(p_true, 1-p_true); abstain if confidence is below tuned threshold.

grounded: p_true = calibrated probability grounded; value = p_true >= 0.5 unless dev tuning indicates otherwise; abstention uses max(p_true, 1-p_true). Do not confuse the classification threshold with the abstention threshold.

All responses include request_id, decision, type, value, confidence/probs or p_true as required, abstain, and measured latency_ms. Follow the provided schema exactly, including all fields.

Appendix B. Dataset distribution snapshot

Task / label

Train n=6500

Dev n=1500

Modeling implication

route: billing

1829 (28.1%)

395 (26.3%)

Largest class; evaluate macro-F1

route: technical

1541 (23.7%)

354 (23.6%)

Confusion with account/general possible

route: account

1324 (20.4%)

298 (19.9%)

Check per-class recall

route: shipping

998 (15.4%)

233 (15.5%)

Minority class; avoid accuracy-only tuning

route: general

808 (12.4%)

220 (14.7%)

Smallest train class

approve: False

4047 (62.3%)

881 (58.7%)

Wrong auto-approval costs 3x

approve: True

2453 (37.7%)

619 (41.3%)

Tune cost-aware decision threshold

grounded: True

3887 (59.8%)

935 (62.3%)

Use reply + article as paired input

grounded: False

2613 (40.2%)

565 (37.7%)

Inspect contradictions/unsupported claims

Source note: counts above are computed directly from the supplied JSONL files. Research-paper summary is a practical synthesis of Guo et al. (2017), cited in the problem statement; confirm paper details against the original publication before quoting externally.