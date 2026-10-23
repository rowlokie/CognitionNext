import sys
sys.path.insert(0, 'src')
import json, pickle
import numpy as np
from pathlib import Path
from baseline import BaselineModel

ROOT = Path(__file__).parent.parent

def main():
    print("=" * 80)
    print("  PHASE 4: ABSTENTION AUDIT (OFFICIAL RANKING vs OPERATIONAL FLAG)")
    print("=" * 80)

    # 1. Load dev data and baseline artifact
    dev_path = ROOT / "data" / "dev.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_records = [json.loads(line) for line in f if line.strip()]

    with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(ROOT / "artifacts" / "thresholds.json", "r", encoding="utf-8") as f:
        thresh_data = json.load(f)

    abstain_thresholds = thresh_data.get("abstention", {})
    decision_thresholds = thresh_data.get("decision", {})

    print(f"Loaded {len(dev_records)} dev records.")
    print(f"Current Configured Abstention Thresholds: {abstain_thresholds}")
    print(f"Current Configured Decision Thresholds:   {decision_thresholds}")

    # Gather predictions and confidences for each task
    tasks = ["route", "approve", "grounded"]
    data = {t: {"confs": [], "errors": [], "preds": [], "golds": [], "abstains": []} for t in tasks}

    for r in dev_records:
        state = r["state"]
        labels = r["labels"]

        # Route
        res_r = model.predict(state, "route")
        conf_r = res_r["confidence"] # max(class probs)
        pred_r = res_r["value"]
        gold_r = labels["route"]
        data["route"]["confs"].append(conf_r)
        data["route"]["preds"].append(pred_r)
        data["route"]["golds"].append(gold_r)
        data["route"]["errors"].append(int(pred_r != gold_r))
        data["route"]["abstains"].append(res_r["abstain"])

        # Approve
        res_a = model.predict(state, "approve")
        p_a = res_a["p_true"]
        conf_a = max(p_a, 1.0 - p_a) # max(p, 1-p)
        pred_a = res_a["value"]
        gold_a = bool(labels["approve"])
        data["approve"]["confs"].append(conf_a)
        data["approve"]["preds"].append(pred_a)
        data["approve"]["golds"].append(gold_a)
        data["approve"]["errors"].append(int(pred_a != gold_a))
        data["approve"]["abstains"].append(res_a["abstain"])

        # Grounded
        res_g = model.predict(state, "grounded")
        p_g = res_g["p_true"]
        conf_g = max(p_g, 1.0 - p_g) # max(p, 1-p)
        pred_g = res_g["value"]
        gold_g = bool(labels["grounded"])
        data["grounded"]["confs"].append(conf_g)
        data["grounded"]["preds"].append(pred_g)
        data["grounded"]["golds"].append(gold_g)
        data["grounded"]["errors"].append(int(pred_g != gold_g))
        data["grounded"]["abstains"].append(res_g["abstain"])

    # 1. Official Confidence Ranking Metric (Top 80% confident, mathematically ignoring abstain flag)
    print("\n" + "=" * 80)
    print("  1. OFFICIAL CONFIDENCE-RANKING METRIC (TOP 80% MOST CONFIDENT)")
    print("     (Slices top 80% of examples ordered by confidence. Strictly ignores abstain flag)")
    print("=" * 80)
    print(f"  {'Task':<12} {'Total N':<10} {'Top 80% N':<12} {'Selective Error':>18} {'Selective Acc':>16}")
    print("  " + "-" * 72)

    official_ranking_results = {}
    for t in tasks:
        confs = np.array(data[t]["confs"])
        errors = np.array(data[t]["errors"])
        n_total = len(confs)
        n_keep = int(n_total * 0.80)
        idx_sorted = np.argsort(confs)[::-1]
        top_errors = errors[idx_sorted[:n_keep]]
        sel_err = float(np.mean(top_errors))
        sel_acc = 1.0 - sel_err
        official_ranking_results[t] = {
            "n_total": n_total,
            "n_keep": n_keep,
            "selective_error": sel_err,
            "selective_acc": sel_acc
        }
        print(f"  {t:<12} {n_total:<10} {n_keep:<12} {sel_err*100:>17.2f}% {sel_acc*100:>15.2f}%")

    # 2. Operational Flag-Based Coverage and Error
    print("\n" + "=" * 80)
    print("  2. OPERATIONAL FLAG-BASED METRICS (USING ACTUAL ABSTAIN BOOLEAN FLAG)")
    print("     (Evaluates coverage and error among answered requests where abstain == False)")
    print("=" * 80)
    print(f"  {'Task':<12} {'Threshold':<12} {'Answered':<10} {'Coverage':>10} {'Abstained':>12} {'Op Error':>12} {'Op Acc':>10}")
    print("  " + "-" * 82)

    operational_results = {}
    for t in tasks:
        th = abstain_thresholds[t]
        abstains = np.array(data[t]["abstains"])
        errors = np.array(data[t]["errors"])
        n_total = len(abstains)
        n_answered = int(np.sum(~abstains))
        n_abstained = int(np.sum(abstains))
        coverage = n_answered / n_total
        op_error = float(np.mean(errors[~abstains])) if n_answered > 0 else 0.0
        op_acc = 1.0 - op_error
        operational_results[t] = {
            "threshold": th,
            "n_answered": n_answered,
            "n_abstained": n_abstained,
            "coverage": coverage,
            "operational_error": op_error,
            "operational_acc": op_acc
        }
        print(f"  {t:<12} {th:<12.4f} {n_answered:<10} {coverage*100:>9.2f}% {n_abstained:>12} {op_error*100:>11.2f}% {op_acc*100:>9.2f}%")

    # 3. Sensitivity Analysis Demonstrating Independence of Ranking Metric from Threshold
    print("\n" + "=" * 80)
    print("  3. THRESHOLD SENSITIVITY vs METRIC INDEPENDENCE AUDIT")
    print("     (Varying threshold shifts operational coverage, but CANNOT alter official ranking error)")
    print("=" * 80)
    sweep_tasks = ["route", "approve", "grounded"]
    sweep_results = {}
    for t in sweep_tasks:
        print(f"\nTask: {t.upper()}")
        print(f"  {'Threshold':<12} {'Op Coverage':>14} {'Op Error (Flag)':>18} {'Official Ranking Error (Top 80%)':>34}")
        print("  " + "-" * 82)
        confs = np.array(data[t]["confs"])
        errors = np.array(data[t]["errors"])
        # Official ranking error is FIXED for the model confidences
        fixed_ranking_err = official_ranking_results[t]["selective_error"]
        
        sweep_results[t] = []
        for test_th in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
            flag_abstain = confs < test_th
            cov = float(np.mean(~flag_abstain))
            op_err = float(np.mean(errors[~flag_abstain])) if np.sum(~flag_abstain) > 0 else 0.0
            sweep_results[t].append({
                "threshold": test_th,
                "coverage": cov,
                "op_error": op_err,
                "ranking_error": fixed_ranking_err
            })
            print(f"  {test_th:<12.2f} {cov*100:>13.2f}% {op_err*100:>17.2f}% {fixed_ranking_err*100:>33.2f}% (UNCHANGED)")

    # Save detailed JSON report
    report_path = ROOT / "reports" / "abstention_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "official_ranking_metrics": official_ranking_results,
            "operational_flag_metrics": operational_results,
            "sensitivity_sweep": sweep_results
        }, f, indent=2)
    print(f"\nSaved abstention audit report to {report_path}")

if __name__ == "__main__":
    main()
