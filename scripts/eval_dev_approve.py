#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import pickle
import json

ROOT = Path(__file__).parent.parent

def eval_approve(model=None):
    if model is None:
        with open(ROOT / "artifacts" / "baseline_model.pkl", "rb") as f:
            model = pickle.load(f)

    with open(ROOT / "artifacts" / "thresholds.json") as f:
        tdata = json.load(f)
    if hasattr(model, "prediction_thresholds") and "prediction" in tdata:
        model.prediction_thresholds.update(tdata["prediction"])
    if hasattr(model, "abstention_thresholds") and "abstention" in tdata:
        model.abstention_thresholds.update(tdata["abstention"])

    dev_path = ROOT / "data" / "dev.jsonl"
    records = [json.loads(line) for line in dev_path.open(encoding="utf-8") if line.strip()]

    tp, fp, fn, tn = 0, 0, 0, 0
    abstains = 0

    for r in records:
        st = r["state"]
        gold = bool(r["labels"]["approve"])
        res = model.predict(st, "approve", r["request_id"])
        pred = res["value"]
        if res["abstain"]:
            abstains += 1
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    cost = 3 * fp + 1 * fn
    acc = (tp + tn) / len(records)
    print(f"Dev Approve: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Cost={cost} (FP={fp}, FN={fn}), Abstains={abstains}/{len(records)} ({abstains/len(records)*100:.1f}%)")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "cost": cost, "fp": fp, "fn": fn, "abstains": abstains}

if __name__ == "__main__":
    eval_approve()
