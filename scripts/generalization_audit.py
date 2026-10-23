import sys
sys.path.insert(0, 'src')
import json, pickle, re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, accuracy_score

from baseline import (
    load_jsonl_safe, clean_text, ROUTE_CLASSES
)

ROOT = Path(__file__).parent.parent

def run_phase1_audit():
    print("=" * 70)
    print("  PHASE 1: MEMORIZATION, LEAKAGE & GENERALIZATION AUDIT")
    print("=" * 70)

    train_path = ROOT / "data" / "train.jsonl"
    dev_path = ROOT / "data" / "dev.jsonl"
    train_records = load_jsonl_safe(str(train_path))
    dev_records = load_jsonl_safe(str(dev_path))

    n_total = len(train_records)
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    n_calib = max(int(n_total * 0.1), 100)
    calib_idx = set(indices[:n_calib])
    fit_idx = set(indices[n_calib:])

    fit_records = [train_records[i] for i in range(n_total) if i in fit_idx]
    calib_records = [train_records[i] for i in range(n_total) if i in calib_idx]

    print(f"Total train rows: {n_total}")
    print(f"Fit slice (90%): {len(fit_records)}")
    print(f"Calib slice (10%): {len(calib_records)}")
    print(f"Dev rows: {len(dev_records)}")
    print(f"Intersection of Fit and Calib indices: {len(fit_idx & calib_idx)}")

    # 1. Check ID leakage
    fit_ids = set(r["request_id"] for r in fit_records)
    calib_ids = set(r["request_id"] for r in calib_records)
    dev_ids = set(r["request_id"] for r in dev_records)

    print(f"\n1. Identifier & ID Overlap:")
    print(f"  Fit vs Calib ID overlap: {len(fit_ids & calib_ids)}")
    print(f"  Fit vs Dev ID overlap: {len(fit_ids & dev_ids)}")
    print(f"  Calib vs Dev ID overlap: {len(calib_ids & dev_ids)}")

    # 2. Check feature extractors in src/baseline.py
    import inspect
    import baseline
    from baseline import BaselineModel, serialize_route, serialize_approve, serialize_grounded
    
    print(f"\n2. Feature Extractor Code Inspection:")
    for fn in [serialize_route, serialize_approve, serialize_grounded, 
               baseline.structured_route, baseline.structured_approve, baseline.structured_grounded]:
        src = inspect.getsource(fn)
        has_id = "request_id" in src
        print(f"  {fn.__name__}: request_id used = {has_id}")

    # 3. Text Duplicate Analysis
    print(f"\n3. Exact & Near-Duplicate Text Analysis:")
    def get_norm_body(r):
        return clean_text(r.get("state", {}).get("body", "")).lower()

    def get_norm_full(r):
        s = r.get("state", {})
        return clean_text(f"{s.get('subject','')} {s.get('body','')}").lower()

    fit_bodies = set(get_norm_body(r) for r in fit_records)
    dev_bodies = [get_norm_body(r) for r in dev_records]
    calib_bodies = set(get_norm_body(r) for r in calib_records)

    exact_body_dev_in_fit = sum(1 for b in dev_bodies if b in fit_bodies)
    exact_body_dev_in_calib = sum(1 for b in dev_bodies if b in calib_bodies)

    fit_fulls = set(get_norm_full(r) for r in fit_records)
    dev_fulls = [get_norm_full(r) for r in dev_records]
    exact_full_dev_in_fit = sum(1 for f in dev_fulls if f in fit_fulls)

    print(f"  Exact dev body in fit slice: {exact_body_dev_in_fit} / {len(dev_records)} ({exact_body_dev_in_fit/len(dev_records)*100:.2f}%)")
    print(f"  Exact dev (subj+body) in fit slice: {exact_full_dev_in_fit} / {len(dev_records)} ({exact_full_dev_in_fit/len(dev_records)*100:.2f}%)")
    print(f"  Exact dev body in calib slice: {exact_body_dev_in_calib} / {len(dev_records)} ({exact_body_dev_in_calib/len(dev_records)*100:.2f}%)")

    # Near duplicate analysis: Jaccard >= 0.85 on 3-grams
    def get_shingles(text, k=3):
        words = text.split()
        if len(words) < k:
            return set([text])
        return set(" ".join(words[i:i+k]) for i in range(len(words)-k+1))

    # Sample check for near duplicates
    fit_shingles_sample = [get_shingles(get_norm_full(r)) for r in fit_records[:1000]]
    dev_shingles_sample = [get_shingles(get_norm_full(r)) for r in dev_records[:500]]

    near_dup_count = 0
    for ds in dev_shingles_sample:
        for fs in fit_shingles_sample:
            if not ds or not fs: continue
            jaccard = len(ds & fs) / len(ds | fs)
            if jaccard >= 0.85:
                near_dup_count += 1
                break
    print(f"  Near-duplicate (subject+body, Jaccard>=0.85) in sample: {near_dup_count} / 500 ({near_dup_count/500*100:.2f}%)")

    # 4. Grouped / Disjoint Template Evaluation
    print(f"\n4. Grouped Performance (Unseen Body Templates vs Repeated Body Templates):")
    # Partition dev into:
    # A) body strictly never seen in train (unique)
    # B) body appeared in train
    dev_unseen_body = [r for r in dev_records if get_norm_body(r) not in fit_bodies]
    dev_seen_body = [r for r in dev_records if get_norm_body(r) in fit_bodies]

    with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
        model = pickle.load(f)

    for group_name, group_records in [("Unseen Body Templates", dev_unseen_body), ("Seen Body Templates", dev_seen_body)]:
        if not group_records: continue
        r_preds, r_golds = [], []
        a_preds, a_golds = [], []
        g_preds, g_golds = [], []
        for r in group_records:
            st = r["state"]
            # route
            res_r = model.predict(st, "route")
            r_preds.append(res_r["value"])
            r_golds.append(r["labels"]["route"])
            # approve
            res_a = model.predict(st, "approve")
            a_preds.append(res_a["value"])
            a_golds.append(r["labels"]["approve"])
            # grounded
            res_g = model.predict(st, "grounded")
            g_preds.append(res_g["value"])
            g_golds.append(r["labels"]["grounded"])

        r_f1 = f1_score(r_golds, r_preds, average="macro")
        r_acc = accuracy_score(r_golds, r_preds)
        a_acc = accuracy_score(a_golds, a_preds)
        a_fp = sum(1 for p, g in zip(a_preds, a_golds) if p and not g)
        a_fn = sum(1 for p, g in zip(a_preds, a_golds) if not p and g)
        a_cost = 3 * a_fp + a_fn
        g_f1 = f1_score(g_golds, g_preds, average="macro")
        g_acc = accuracy_score(g_golds, g_preds)

        print(f"  Group: {group_name} (N={len(group_records)}):")
        print(f"    Route Macro-F1: {r_f1*100:.2f}%, Acc: {r_acc*100:.2f}%")
        print(f"    Approve Cost: {a_cost} (FP={a_fp}, FN={a_fn}), Acc: {a_acc*100:.2f}%")
        print(f"    Grounded Macro-F1: {g_f1*100:.2f}%, Acc: {g_acc*100:.2f}%")

    # 5. Generalization Gap: Train (Fit Slice) vs Dev Set Metrics
    print(f"\n5. Generalization Gap (Fit Slice N=5,850 vs Dev Set N=1,500):")
    # Sample 1500 fit records to compare fairly
    fit_sample = fit_records[:1500]

    def eval_records(records, name):
        r_preds, r_golds = [], []
        a_preds, a_golds = [], []
        g_preds, g_golds = [], []
        for r in records:
            st = r["state"]
            res_r = model.predict(st, "route")
            r_preds.append(res_r["value"])
            r_golds.append(r["labels"]["route"])

            res_a = model.predict(st, "approve")
            a_preds.append(res_a["value"])
            a_golds.append(r["labels"]["approve"])

            res_g = model.predict(st, "grounded")
            g_preds.append(res_g["value"])
            g_golds.append(r["labels"]["grounded"])

        r_f1 = f1_score(r_golds, r_preds, average="macro")
        r_acc = accuracy_score(r_golds, r_preds)

        a_fp = sum(1 for p, g in zip(a_preds, a_golds) if p and not g)
        a_fn = sum(1 for p, g in zip(a_preds, a_golds) if not p and g)
        a_cost = 3 * a_fp + a_fn
        a_acc = accuracy_score(a_golds, a_preds)

        g_f1 = f1_score(g_golds, g_preds, average="macro")
        g_acc = accuracy_score(g_golds, g_preds)

        return {
            "name": name,
            "route_macro_f1": r_f1,
            "route_acc": r_acc,
            "approve_cost": a_cost,
            "approve_fp": a_fp,
            "approve_fn": a_fn,
            "approve_acc": a_acc,
            "grounded_macro_f1": g_f1,
            "grounded_acc": g_acc,
        }

    fit_m = eval_records(fit_sample, "Train Fit (1,500 sample)")
    dev_m = eval_records(dev_records, "Dev Set (1,500 all)")

    print(f"  {'Metric':<25} {'Train (Fit)':>15} {'Dev':>15} {'Gap (Train - Dev)':>20}")
    print("  " + "-" * 75)
    print(f"  {'Route Macro-F1':<25} {fit_m['route_macro_f1']*100:>14.2f}% {dev_m['route_macro_f1']*100:>14.2f}% {(fit_m['route_macro_f1']-dev_m['route_macro_f1'])*100:>19.2f}%")
    print(f"  {'Route Accuracy':<25} {fit_m['route_acc']*100:>14.2f}% {dev_m['route_acc']*100:>14.2f}% {(fit_m['route_acc']-dev_m['route_acc'])*100:>19.2f}%")
    print(f"  {'Approve Accuracy':<25} {fit_m['approve_acc']*100:>14.2f}% {dev_m['approve_acc']*100:>14.2f}% {(fit_m['approve_acc']-dev_m['approve_acc'])*100:>19.2f}%")
    print(f"  {'Approve Cost (3*FP+FN)':<25} {fit_m['approve_cost']:>15d} {dev_m['approve_cost']:>15d} {fit_m['approve_cost']-dev_m['approve_cost']:>20d}")
    print(f"  {'Grounded Macro-F1':<25} {fit_m['grounded_macro_f1']*100:>14.2f}% {dev_m['grounded_macro_f1']*100:>14.2f}% {(fit_m['grounded_macro_f1']-dev_m['grounded_macro_f1'])*100:>19.2f}%")
    print(f"  {'Grounded Accuracy':<25} {fit_m['grounded_acc']*100:>14.2f}% {dev_m['grounded_acc']*100:>14.2f}% {(fit_m['grounded_acc']-dev_m['grounded_acc'])*100:>19.2f}%")

if __name__ == "__main__":
    run_phase1_audit()
