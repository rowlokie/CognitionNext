"""
Promote the enriched grounded candidate to production.
Steps:
  1. Load baseline_model.pkl
  2. Load grounded_candidate_v2.pkl
  3. Swap in the new grounded_model and threshold
  4. Save back to baseline_model.pkl (after backing up the original)
  5. Confirm route/approve heads are untouched
"""
import sys, pickle, shutil, json
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np

ROOT = Path(".")
ARTIFACTS = ROOT / "artifacts"

# ---- 1. Load artifacts ---------------------------------------------------
prod_path = ARTIFACTS / "baseline_model.pkl"
cand_path = ARTIFACTS / "grounded_candidate_v2.pkl"
backup_path = ARTIFACTS / "baseline_model_v1_backup.pkl"

print("Loading production baseline...")
with open(prod_path, "rb") as f:
    prod = pickle.load(f)

print("Loading grounded candidate v2...")
with open(cand_path, "rb") as f:
    cand = pickle.load(f)

# ---- 2. Back up the original (idempotent: skip if backup already exists) ---
if not backup_path.exists():
    shutil.copy2(prod_path, backup_path)
    print(f"Original backed up to {backup_path}")
else:
    print(f"Backup already exists at {backup_path} — skipping")

# ---- 3. Swap the grounded head -------------------------------------------
old_threshold = prod.prediction_thresholds.get("grounded", 0.51)
prod.grounded_model = cand["grounded_model"]
prod.prediction_thresholds["grounded"] = float(cand["threshold"])

print(f"\nSwapped grounded head:")
print(f"  Old threshold: {old_threshold:.3f}")
print(f"  New threshold: {prod.prediction_thresholds['grounded']:.3f}")
print(f"  Candidate macro-F1: {cand['macro_f1']*100:.2f}%")
print(f"  Candidate ECE:      {cand['ece']:.4f}")
print(f"  Candidate BSS:      {cand['bss']:.4f}")
print(f"  Feature set:        {cand['feature_set']}")

# Confirm route and approve heads are the original objects (not modified)
print(f"\nroute_model type:   {type(prod.route_model).__name__}")
print(f"approve_model type: {type(prod.approve_model).__name__}")
print(f"grounded_model type:{type(prod.grounded_model).__name__}")

# ---- 4. Save promoted production artifact --------------------------------
with open(prod_path, "wb") as f:
    pickle.dump(prod, f)
print(f"\nProduction artifact saved: {prod_path}")

# ---- 5. Update results.json -------------------------------------------
results_path = ROOT / "results.json"
if results_path.exists():
    with open(results_path, "r") as f:
        results = json.load(f)
else:
    results = {}

results.update({
    "grounded_macro_f1_candidate": round(cand['macro_f1'] * 100, 2),
    "grounded_ece_candidate": round(cand['ece'], 4),
    "grounded_bss_candidate": round(cand['bss'], 4),
    "grounded_threshold_promoted": round(float(cand["threshold"]), 3),
    "grounded_feature_version": "enriched_v2_23features",
})

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print("results.json updated.")
print("\n[DONE] Promotion complete. Restart the server to load the new artifact.")
