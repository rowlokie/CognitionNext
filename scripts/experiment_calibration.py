import sys
sys.path.insert(0, 'src')
import json, pickle, re, time, math
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack

from baseline import (
    load_jsonl_safe, clean_text, safe_str, tier_onehot, amount_features,
    serialize_route, serialize_approve, serialize_grounded,
    structured_route, structured_approve, structured_grounded,
    BaselineModel, ROUTE_CLASSES
)

ROOT = Path(__file__).parent.parent

def compute_ece(probs_conf, corrects, n_bins=15):
    """
    Computes ECE using 15 equal-width bins on [0, 1].
    probs_conf: 1D array of prediction confidences (max prob for route, or max(p, 1-p), or p_true)
    corrects: 1D boolean/binary array indicating whether prediction matched gold
    """
    probs_conf = np.asarray(probs_conf)
    corrects = np.asarray(corrects, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs_conf)
    ece = 0.0
    bin_summaries = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs_conf >= lo) & (probs_conf <= hi)
        else:
            mask = (probs_conf >= lo) & (probs_conf < hi)
        
        count = int(mask.sum())
        if count > 0:
            avg_conf = float(np.mean(probs_conf[mask]))
            emp_acc = float(np.mean(corrects[mask]))
            gap = abs(avg_conf - emp_acc)
            weight = count / total
            ece += weight * gap
        else:
            avg_conf = 0.0
            emp_acc = 0.0
            gap = 0.0
            weight = 0.0
        
        bin_summaries.append({
            "bin": i + 1,
            "range": f"[{lo:.2f}, {hi:.2f}" + ("]" if i == n_bins - 1 else ")"),
            "count": count,
            "pct": (count / total) * 100 if total > 0 else 0.0,
            "avg_conf": avg_conf,
            "emp_acc": emp_acc,
            "gap": gap,
            "weighted_gap": weight * gap
        })

    return float(ece), bin_summaries

def compute_brier_and_skill_binary(probs, golds):
    probs = np.asarray(probs, dtype=float)
    golds = np.asarray(golds, dtype=float)
    base_rate = float(np.mean(golds))
    bs = float(np.mean((probs - golds) ** 2))
    bs_ref = float(np.mean((base_rate - golds) ** 2))
    bss = 1.0 - (bs / (bs_ref + 1e-12))
    return bs, bss

def compute_brier_and_skill_multiclass(probs_matrix, golds_indices, n_classes=5):
    probs_matrix = np.asarray(probs_matrix, dtype=float)
    golds_indices = np.asarray(golds_indices, dtype=int)
    n = len(golds_indices)
    one_hot = np.zeros((n, n_classes), dtype=float)
    for i, g in enumerate(golds_indices):
        one_hot[i, g] = 1.0
    
    priors = np.mean(one_hot, axis=0, keepdims=True)
    bs = float(np.mean(np.sum((probs_matrix - one_hot) ** 2, axis=1)))
    bs_ref = float(np.mean(np.sum((priors - one_hot) ** 2, axis=1)))
    bss = 1.0 - (bs / (bs_ref + 1e-12))
    return bs, bss

def main():
    print("=" * 80)
    print("  PHASE 3: RIGOROUS CALIBRATION AUDIT (ISOTONIC vs SIGMOID / PLATT)")
    print("=" * 80)

    train_path = ROOT / "data" / "train.jsonl"
    dev_path = ROOT / "data" / "dev.jsonl"
    train_records = load_jsonl_safe(str(train_path))
    dev_records = load_jsonl_safe(str(dev_path))

    # Split train into fit (5,850) and calib (650)
    np.random.seed(42)
    n_total = len(train_records)
    n_calib = max(int(n_total * 0.1), 100)
    indices = np.random.permutation(n_total)
    calib_idx = set(indices[:n_calib])
    fit_records = [train_records[i] for i in range(n_total) if i not in calib_idx]
    calib_records = [train_records[i] for i in range(n_total) if i in calib_idx]

    print(f"Sample Counts:")
    print(f"  Training Fit Set:         {len(fit_records):,} rows (used for TF-IDF & base classifiers)")
    print(f"  Calibration Slice:        {len(calib_records):,} rows (held-out for threshold tuning)")
    print(f"  Dev Validation Set:       {len(dev_records):,} rows (held-out evaluation)")
    print(f"  Cross-Validation Folds:   3 folds (each fold: 3,900 train, 1,950 calib)")

    # Load baseline model for shared TF-IDF
    with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
        baseline_model = pickle.load(f)

    shared_word = baseline_model.shared_tfidf_word
    shared_char = baseline_model.shared_tfidf_char

    results = {}

    tasks = ["route", "approve", "grounded"]

    for task in tasks:
        print(f"\n" + "-" * 80)
        print(f"  EVALUATING TASK: {task.upper()}")
        print("-" * 80)

        if task == "route":
            fit_texts = [serialize_route(r["state"]) for r in fit_records]
            fit_structs = [structured_route(r["state"]) for r in fit_records]
            dev_texts = [serialize_route(r["state"]) for r in dev_records]
            dev_structs = [structured_route(r["state"]) for r in dev_records]
            y_fit = np.array([r["labels"]["route"] for r in fit_records])
            y_dev = np.array([r["labels"]["route"] for r in dev_records])
            class_map = {c: i for i, c in enumerate(ROUTE_CLASSES)}
            y_fit_num = np.array([class_map[c] for c in y_fit])
            y_dev_num = np.array([class_map[c] for c in y_dev])
            base_clf = LogisticRegression(C=2.0, max_iter=600, class_weight="balanced", solver="lbfgs")
        elif task == "approve":
            fit_texts = [serialize_approve(r["state"]) for r in fit_records]
            fit_structs = [structured_approve(r["state"]) for r in fit_records]
            dev_texts = [serialize_approve(r["state"]) for r in dev_records]
            dev_structs = [structured_approve(r["state"]) for r in dev_records]
            y_fit_num = np.array([int(bool(r["labels"]["approve"])) for r in fit_records])
            y_dev_num = np.array([int(bool(r["labels"]["approve"])) for r in dev_records])
            base_clf = LogisticRegression(C=1.0, max_iter=600, class_weight="balanced", solver="lbfgs")
        else: # grounded
            fit_texts = [serialize_grounded(r["state"]) for r in fit_records]
            fit_structs = [structured_grounded(r["state"]) for r in fit_records]
            dev_texts = [serialize_grounded(r["state"]) for r in dev_records]
            dev_structs = [structured_grounded(r["state"]) for r in dev_records]
            y_fit_num = np.array([int(bool(r["labels"]["grounded"])) for r in fit_records])
            y_dev_num = np.array([int(bool(r["labels"]["grounded"])) for r in dev_records])
            base_clf = LogisticRegression(C=1.0, max_iter=600, class_weight="balanced", solver="lbfgs")

        X_fit = baseline_model._combine_features(fit_texts, fit_structs, shared_word, shared_char, fit=False)
        X_dev = baseline_model._combine_features(dev_texts, dev_structs, shared_word, shared_char, fit=False)

        # 1. Uncalibrated
        t0 = time.perf_counter()
        base_clf.fit(X_fit, y_fit_num)
        probs_uncal = base_clf.predict_proba(X_dev)
        lat_uncal = ((time.perf_counter() - t0) / len(dev_records)) * 1000

        # 2. Sigmoid (Platt) 3-fold CV
        t0 = time.perf_counter()
        sig_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        sig_clf.fit(X_fit, y_fit_num)
        probs_sig = sig_clf.predict_proba(X_dev)
        lat_sig = ((time.perf_counter() - t0) / len(dev_records)) * 1000

        # 3. Isotonic 3-fold CV
        t0 = time.perf_counter()
        iso_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        iso_clf.fit(X_fit, y_fit_num)
        probs_iso = iso_clf.predict_proba(X_dev)
        lat_iso = ((time.perf_counter() - t0) / len(dev_records)) * 1000

        # Compute metrics for each
        task_res = {}
        for name, probs in [("Uncalibrated", probs_uncal), ("Sigmoid (Platt)", probs_sig), ("Isotonic (Current)", probs_iso)]:
            if task == "route":
                preds = np.argmax(probs, axis=1)
                confs = np.max(probs, axis=1)
                corrects = (preds == y_dev_num)
                ece, bins = compute_ece(confs, corrects, n_bins=15)
                bs, bss = compute_brier_and_skill_multiclass(probs, y_dev_num, n_classes=5)
                f1 = f1_score(y_dev_num, preds, average="macro")
                acc = accuracy_score(y_dev_num, preds)
                task_res[name] = {
                    "ece": ece, "bs": bs, "bss": bss, "f1": f1, "acc": acc, "bins": bins
                }
            else: # binary
                p_pos = probs[:, 1]
                # Default 0.5 decision threshold for comparison
                preds = (p_pos >= 0.50).astype(int)
                confs = np.maximum(p_pos, 1.0 - p_pos)
                corrects = (preds == y_dev_num)
                # Official binary ECE: confidence is confs, accuracy is corrects
                ece, bins = compute_ece(confs, corrects, n_bins=15)
                # Also probability calibration on positive class
                ece_pos, _ = compute_ece(p_pos, y_dev_num, n_bins=15)
                bs, bss = compute_brier_and_skill_binary(p_pos, y_dev_num)
                f1 = f1_score(y_dev_num, preds, average="macro")
                acc = accuracy_score(y_dev_num, preds)
                task_res[name] = {
                    "ece": ece, "ece_pos": ece_pos, "bs": bs, "bss": bss, "f1": f1, "acc": acc, "bins": bins
                }

        results[task] = task_res

        # Print summary table for this task
        print(f"  {'Method':<20} {'ECE (15-bin)':>14} {'Brier Score':>14} {'Brier Skill':>14} {'Macro-F1':>12} {'Accuracy':>12}")
        print("  " + "-" * 76)
        for name in ["Uncalibrated", "Sigmoid (Platt)", "Isotonic (Current)"]:
            r = task_res[name]
            print(f"  {name:<20} {r['ece']:>14.4f} {r['bs']:>14.4f} {r['bss']:>14.4f} {r['f1']*100:>11.2f}% {r['acc']*100:>11.2f}%")

    # Reliability Bin Summaries for Isotonic vs Sigmoid on Dev
    print("\n" + "=" * 80)
    print("  RELIABILITY DIAGRAM BIN SUMMARIES (15 Equal-Width Bins) ON DEV")
    print("=" * 80)
    for task in tasks:
        print(f"\nTask: {task.upper()} (Current Isotonic vs Sigmoid)")
        print(f"  {'Bin':<4} {'Confidence Range':<16} {'Count':>6} | {'Iso AvgConf':>11} {'Iso EmpAcc':>11} {'Iso Gap':>9} | {'Sig AvgConf':>11} {'Sig EmpAcc':>11} {'Sig Gap':>9}")
        print("  " + "-" * 88)
        iso_bins = results[task]["Isotonic (Current)"]["bins"]
        sig_bins = results[task]["Sigmoid (Platt)"]["bins"]
        for b_iso, b_sig in zip(iso_bins, sig_bins):
            if b_iso["count"] > 0 or b_sig["count"] > 0:
                print(f"  {b_iso['bin']:<4} {b_iso['range']:<16} {b_iso['count']:>6} | {b_iso['avg_conf']:>11.3f} {b_iso['emp_acc']:>11.3f} {b_iso['gap']:>9.3f} | {b_sig['avg_conf']:>11.3f} {b_sig['emp_acc']:>11.3f} {b_sig['gap']:>9.3f}")

    # Save summary
    out_path = ROOT / "reports" / "calibration_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    clean_res = {}
    for t in tasks:
        clean_res[t] = {}
        for m in results[t]:
            clean_res[t][m] = {k: v for k, v in results[t][m].items() if k != "bins"}
    with open(out_path, "w") as f:
        json.dump(clean_res, f, indent=2)
    print(f"\nSaved calibration comparison metrics to {out_path}")

if __name__ == "__main__":
    main()
