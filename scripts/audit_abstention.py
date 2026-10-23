import sys
sys.path.insert(0, 'src')
import json, pickle, numpy as np

with open('data/dev.jsonl', 'r', encoding='utf-8') as f:
    dev = [json.loads(line) for line in f if line.strip()]

with open('artifacts/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/thresholds.json', 'r') as f:
    thresh_data = json.load(f)

abstain_thresh = thresh_data['abstention']
print('Abstention thresholds from artifacts/thresholds.json:', abstain_thresh)

# 1. Route
route_confs = []
route_errors = []
route_abstains = []
for r in dev:
    res = model.predict(r['state'], 'route', r.get('request_id', ''))
    conf = res['confidence']
    pred = res['value']
    gold = r['labels']['route']
    err = int(pred != gold)
    route_confs.append(conf)
    route_errors.append(err)
    route_abstains.append(conf < abstain_thresh['route'])

# 2. Approve (test both at 0.50 and 0.730)
app_confs = []
app_errors_05 = []
app_errors_073 = []
app_abstains = []
for r in dev:
    res = model.predict(r['state'], 'approve', r.get('request_id', ''))
    p = res['p_true']
    conf = max(p, 1 - p)
    pred_05 = p >= 0.50
    pred_073 = p >= 0.730
    gold = bool(r['labels']['approve'])
    app_confs.append(conf)
    app_errors_05.append(int(pred_05 != gold))
    app_errors_073.append(int(pred_073 != gold))
    app_abstains.append(conf < abstain_thresh['approve'])

# 3. Grounded
gr_confs = []
gr_errors = []
gr_abstains = []
for r in dev:
    res = model.predict(r['state'], 'grounded', r.get('request_id', ''))
    p = res['p_true']
    conf = max(p, 1 - p)
    pred = p >= 0.50
    gold = bool(r['labels']['grounded'])
    gr_confs.append(conf)
    gr_errors.append(int(pred != gold))
    gr_abstains.append(conf < abstain_thresh['grounded'])

def compute_selective_error(confs, errors, coverage=0.80):
    n_keep = int(len(confs) * coverage)
    idx = np.argsort(confs)[::-1]
    top_errs = np.array(errors)[idx[:n_keep]]
    return float(np.mean(top_errs))

print('\n=== Selective Error on Most Confident 80% (Official Definition) ===')
r_sel = compute_selective_error(route_confs, route_errors, 0.80)
a_sel_05 = compute_selective_error(app_confs, app_errors_05, 0.80)
a_sel_073 = compute_selective_error(app_confs, app_errors_073, 0.80)
g_sel = compute_selective_error(gr_confs, gr_errors, 0.80)

print(f'Route selective error (top 80%): {r_sel:.4f} ({r_sel*100:.2f}%)')
print(f'Approve selective error (pred>=0.50, top 80%): {a_sel_05:.4f} ({a_sel_05*100:.2f}%)')
print(f'Approve selective error (pred>=0.73, top 80%): {a_sel_073:.4f} ({a_sel_073*100:.2f}%)')
print(f'Grounded selective error (top 80%): {g_sel:.4f} ({g_sel*100:.2f}%)')

print('\n=== Operational Coverage on Dev using Saved Abstention Thresholds ===')
print('Route coverage: %.2f%%' % ((1 - np.mean(route_abstains)) * 100))
print('Approve coverage: %.2f%%' % ((1 - np.mean(app_abstains)) * 100))
print('Grounded coverage: %.2f%%' % ((1 - np.mean(gr_abstains)) * 100))
