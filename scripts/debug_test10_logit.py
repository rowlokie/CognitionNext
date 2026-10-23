import sys, pickle
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import numpy as np
from scipy.sparse import hstack, csr_matrix
from exp_interaction_train import cand_clf, build_structured, shared_word, shared_char, serialize_grounded

state10 = {"customer_tier": "pro", "amount": 0.0, "draft_reply": "Standard warranty lasts 24 months.", "kb_article": "Standard warranty lasts 12 months."}
t = serialize_grounded(state10)
s = build_structured(state10)
x = hstack([shared_word.transform([t]), shared_char.transform([t]), csr_matrix([s])])

lr = cand_clf.calibrated_classifiers_[0].estimator
logit = lr.decision_function(x)[0]
p = cand_clf.predict_proba(x)[0, 1]
print(f"Test 10 raw logit: {logit:.4f}, calibrated p_true: {p:.4f}")

# What contributed to the logit?
w_struct = lr.coef_[0, -len(s):]
struct_contrib = np.sum(w_struct * s)
print(f"Structured features contribution: {struct_contrib:.4f}")
print(f"TF-IDF features contribution: {logit - struct_contrib - lr.intercept_[0]:.4f}")
print(f"Intercept: {lr.intercept_[0]:.4f}")
