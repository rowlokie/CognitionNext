import sys
sys.path.insert(0, 'src')
import json, pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

from baseline import BaselineModel, ROUTE_CLASSES

ROOT = Path(__file__).parent.parent

def compute_ece(probs_conf, corrects, n_bins=15):
    probs_conf = np.asarray(probs_conf)
    corrects = np.asarray(corrects, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs_conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs_conf >= lo) & (probs_conf <= hi) if i == n_bins - 1 else (probs_conf >= lo) & (probs_conf < hi)
        count = int(mask.sum())
        if count > 0:
            avg_conf = float(np.mean(probs_conf[mask]))
            emp_acc = float(np.mean(corrects[mask]))
            ece += (count / total) * abs(avg_conf - emp_acc)
    return float(ece)

def compute_brier_skill_binary(probs, golds):
    probs = np.asarray(probs, dtype=float)
    golds = np.asarray(golds, dtype=float)
    base_rate = float(np.mean(golds))
    bs = float(np.mean((probs - golds) ** 2))
    bs_ref = float(np.mean((base_rate - golds) ** 2))
    bss = 1.0 - (bs / (bs_ref + 1e-12))
    return bs, max(0.0, bss)

def compute_brier_skill_multiclass(probs_matrix, golds_indices, n_classes=5):
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
    return bs, max(0.0, bss)

def main():
    dev_path = ROOT / "data" / "dev.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev = [json.loads(line) for line in f if line.strip()]

    with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(ROOT / "artifacts" / "thresholds.json", "r", encoding="utf-8") as f:
        tdata = json.load(f)

    # Predictions
    # Route
    route_preds = []
    route_confs = []
    route_probs = []
    route_golds = []
    class_to_idx = {c: i for i, c in enumerate(ROUTE_CLASSES)}
    for r in dev:
        res = model.predict(r["state"], "route")
        route_preds.append(res["value"])
        route_confs.append(res["confidence"])
        p_vec = [res["probs"][c] for c in ROUTE_CLASSES]
        route_probs.append(p_vec)
        route_golds.append(r["labels"]["route"])

    route_golds_idx = [class_to_idx[g] for g in route_golds]
    route_preds_idx = [class_to_idx[p] for p in route_preds]
    route_f1 = f1_score(route_golds, route_preds, average="macro")
    route_corrects = (np.array(route_preds) == np.array(route_golds))
    route_ece = compute_ece(route_confs, route_corrects)
    route_bs, route_bss = compute_brier_skill_multiclass(route_probs, route_golds_idx)

    # Approve
    app_probs = []
    app_preds = []
    app_golds = []
    app_confs = []
    # Optimal threshold from thresholds.json is 0.700
    app_th = tdata.get("prediction", {}).get("approve", 0.70)
    for r in dev:
        res = model.predict(r["state"], "approve")
        p = res["p_true"]
        app_probs.append(p)
        pred = (p >= app_th)
        app_preds.append(pred)
        app_confs.append(max(p, 1.0 - p))
        app_golds.append(bool(r["labels"]["approve"]))

    app_golds = np.array(app_golds)
    app_preds = np.array(app_preds)
    fp = np.sum(app_preds & ~app_golds)
    fn = np.sum(~app_preds & app_golds)
    app_cost = 3 * fp + fn
    # Refusing every request (all False):
    cost_refuse_all = 3 * 0 + np.sum(app_golds) # 619
    # Mapping onto 0-1 score in which refusing every request scores 0.25:
    # Score = 1.0 - (1.0 - 0.25) * (cost / cost_refuse_all)
    # If cost = 0, score = 1.0; if cost = 619, score = 0.25.
    app_score_01 = max(0.0, 1.0 - 0.75 * (app_cost / cost_refuse_all))
    app_corrects = (app_preds == app_golds)
    app_ece = compute_ece(app_confs, app_corrects)
    app_bs, app_bss = compute_brier_skill_binary(app_probs, app_golds)

    # Grounded
    gr_probs = []
    gr_preds = []
    gr_golds = []
    gr_confs = []
    gr_th = tdata.get("prediction", {}).get("grounded", 0.51)
    for r in dev:
        res = model.predict(r["state"], "grounded")
        p = res["p_true"]
        gr_probs.append(p)
        pred = (p >= gr_th)
        gr_preds.append(pred)
        gr_confs.append(max(p, 1.0 - p))
        gr_golds.append(bool(r["labels"]["grounded"]))

    gr_golds = np.array(gr_golds)
    gr_preds = np.array(gr_preds)
    gr_f1 = f1_score(gr_golds, gr_preds, average="macro")
    gr_corrects = (gr_preds == gr_golds)
    gr_ece = compute_ece(gr_confs, gr_corrects)
    gr_bs, gr_bss = compute_brier_skill_binary(gr_probs, gr_golds)

    # -------------------------------------------------------------
    # 1. QUALITY (40 points)
    # Average of route macro-F1, approve mapped score, grounded macro-F1
    # -------------------------------------------------------------
    q_route = route_f1
    q_approve = app_score_01
    q_grounded = gr_f1
    quality_avg = (q_route + q_approve + q_grounded) / 3.0
    quality_points = quality_avg * 40.0

    # -------------------------------------------------------------
    # 2. CALIBRATION (25 points)
    # Half ECE, half Brier skill
    # ECE score: 1.0 - ECE (lower ECE is better)
    # BSS: direct skill score [0, 1]
    # -------------------------------------------------------------
    avg_ece = (route_ece + app_ece + gr_ece) / 3.0
    # ECE score component: perfectly calibrated (ECE=0) gives 1.0, ECE=0.2 gives 0.0
    ece_component = max(0.0, 1.0 - (avg_ece / 0.20))
    avg_bss = (route_bss + app_bss + gr_bss) / 3.0
    bss_component = avg_bss
    calib_avg = 0.5 * ece_component + 0.5 * bss_component
    calib_points = calib_avg * 25.0

    # -------------------------------------------------------------
    # 3. LATENCY (20 points)
    # full marks at <= 25ms, scaling to 0 at 100ms
    # -------------------------------------------------------------
    with open(ROOT / "results.json", "r") as f:
        bench_data = json.load(f)
    p95_route = bench_data["latency"]["route"]["p95_ms"]
    p95_app = bench_data["latency"]["approve"]["p95_ms"]
    p95_gr = bench_data["latency"]["grounded"]["p95_ms"]
    p95_avg = (p95_route + p95_app + p95_gr) / 3.0

    if p95_avg <= 25.0:
        latency_fraction = 1.0
    elif p95_avg >= 100.0:
        latency_fraction = 0.0
    else:
        latency_fraction = (100.0 - p95_avg) / (100.0 - 25.0)
    latency_points = latency_fraction * 20.0

    # -------------------------------------------------------------
    # 4. ABSTENTION (15 points)
    # Error rate on cases chosen to answer, with coverage fixed at 80%
    # (Evaluated on the top 80% most confident cases)
    # -------------------------------------------------------------
    def get_top80_acc(confs, corrects):
        confs = np.array(confs)
        corrects = np.array(corrects)
        n_keep = int(len(confs) * 0.80)
        idx = np.argsort(confs)[::-1][:n_keep]
        err_rate = 1.0 - np.mean(corrects[idx])
        acc_rate = np.mean(corrects[idx])
        return err_rate, acc_rate

    route_sel_err, route_sel_acc = get_top80_acc(route_confs, route_corrects)
    app_sel_err, app_sel_acc = get_top80_acc(app_confs, app_corrects)
    gr_sel_err, gr_sel_acc = get_top80_acc(gr_confs, gr_corrects)

    avg_sel_err = (route_sel_err + app_sel_err + gr_sel_err) / 3.0
    avg_sel_acc = (route_sel_acc + app_sel_acc + gr_sel_acc) / 3.0

    # If abstention score is based on accuracy on the top 80% (or 1 - error scaled against unselective baseline):
    # Standard: score = (1.0 - avg_sel_err / baseline_err) or selective accuracy
    # In the problem statement: "your error rate on the cases you chose to answer, with coverage fixed at 80%"
    # Error rate of 0% gives 1.0, 50% gives 0.0:
    abstention_fraction_acc = avg_sel_acc # raw accuracy
    abstention_fraction_err = max(0.0, 1.0 - (avg_sel_err / 0.25)) # if 25% error is zero floor

    abstention_points_1 = abstention_fraction_acc * 15.0
    abstention_points_2 = abstention_fraction_err * 15.0

    print("=" * 70)
    print("  EXACT MEASURED METRICS & POINTS BREAKDOWN (OUT OF 100)")
    print("=" * 70)
    print(f"1. QUALITY (Max 40.0 pts):")
    print(f"   - Route Macro-F1:            {route_f1:.4f} ({route_f1*100:.2f}%)")
    print(f"   - Approve Cost (3*FP+FN):     {app_cost} (FP={fp}, FN={fn})")
    print(f"   - Approve Mapped Score:       {app_score_01:.4f} (refuse all cost={cost_refuse_all} -> 0.25)")
    print(f"   - Grounded Macro-F1:         {gr_f1:.4f} ({gr_f1*100:.2f}%)")
    print(f"   - Quality Average (0-1):     {quality_avg:.4f}")
    print(f"   => QUALITY POINTS:           {quality_points:.2f} / 40.00")
    print()
    print(f"2. CALIBRATION (Max 25.0 pts):")
    print(f"   - Route:    ECE={route_ece:.4f}, BSS={route_bss:.4f}")
    print(f"   - Approve:  ECE={app_ece:.4f}, BSS={app_bss:.4f}")
    print(f"   - Grounded: ECE={gr_ece:.4f}, BSS={gr_bss:.4f}")
    print(f"   - Average ECE:               {avg_ece:.4f} (Score={ece_component:.4f})")
    print(f"   - Average BSS:               {avg_bss:.4f}")
    print(f"   - Calibration Average (0-1): {calib_avg:.4f}")
    print(f"   => CALIBRATION POINTS:       {calib_points:.2f} / 25.00")
    print()
    print(f"3. LATENCY (Max 20.0 pts):")
    print(f"   - Route p95:                 {p95_route:.2f} ms")
    print(f"   - Approve p95:               {p95_app:.2f} ms")
    print(f"   - Grounded p95:              {p95_gr:.2f} ms")
    print(f"   - Average p95:               {p95_avg:.2f} ms (<=25ms target)")
    print(f"   => LATENCY POINTS:           {latency_points:.2f} / 20.00 (FULL MARKS)")
    print()
    print(f"4. ABSTENTION (Max 15.0 pts):")
    print(f"   - Route Selective Error:     {route_sel_err*100:.2f}% (Acc={route_sel_acc*100:.2f}%)")
    print(f"   - Approve Selective Error:   {app_sel_err*100:.2f}% (Acc={app_sel_acc*100:.2f}%)")
    print(f"   - Grounded Selective Error:  {gr_sel_err*100:.2f}% (Acc={gr_sel_acc*100:.2f}%)")
    print(f"   - Average Selective Error:   {avg_sel_err*100:.2f}% (Average Acc={avg_sel_acc*100:.2f}%)")
    print(f"   => If scored by Selective Acc: {abstention_points_1:.2f} / 15.00")
    print(f"   => If scaled vs error floor:   {abstention_points_2:.2f} / 15.00")
    print()
    print("=" * 70)
    print(f"TOTAL ESTIMATED SCORE:          {quality_points + calib_points + latency_points + abstention_points_2:.2f} - {quality_points + calib_points + latency_points + abstention_points_1:.2f} / 100")
    print("=" * 70)

if __name__ == "__main__":
    main()
