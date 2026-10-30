"""Stage 6: simulate production traffic and watch for DATA drift (PSI) and FAIRNESS drift.

Scenario: a new acquisition channel brings in more applicants under 30, and those
applicants have progressively weaker bureau scores (EXT_SOURCE). Labels are not
available in production yet, so fairness is tracked with the label-free metric:
the gap in rejection rate between groups (demographic parity).
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import PARAMS, PROC, MODELS, REPORTS, NUMERIC, ENGINEERED, save_json
from predictor import predict

def psi(ref, cur, bins=10):
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref); c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))

def make_batch(te, k, rng, cfg):
    young_share = 0.2 + 0.05 * k
    young, rest = te[te.age_band == "<30"], te[te.age_band != "<30"]
    n_y = int(cfg["batch_size"] * young_share)
    b = pd.concat([young.sample(n_y, replace=True, random_state=int(rng.integers(1e9))),
                   rest.sample(cfg["batch_size"] - n_y, replace=True, random_state=int(rng.integers(1e9)))])
    m = b.age_band == "<30"
    for c in ("EXT_SOURCE_2", "EXT_SOURCE_3"):
        b.loc[m, c] = (b.loc[m, c] - 0.035 * k).clip(0, 1)
    return b

def main():
    cfg = PARAMS["monitor"]; rng = np.random.default_rng(1)
    bundle = joblib.load(MODELS / "production.joblib")
    ref = pd.read_parquet(PROC / "train.parquet"); te = pd.read_parquet(PROC / "test.parquet")
    feats = NUMERIC + ENGINEERED
    rows = []
    for k in range(cfg["n_batches"]):
        b = make_batch(te, k, rng, cfg)
        _, dec = predict(bundle, b, seed=k)
        b = b.assign(rejected=dec)
        rate = {col: b.groupby(col)["rejected"].mean() for col in ("gender", "age_band")}
        gap = max(float(v.max() - v.min()) for v in rate.values())
        psis = {f: psi(ref[f], b[f]) for f in feats}
        top = max(psis, key=psis.get)
        rows.append({"batch": k + 1, "share_under_30": round(float((b.age_band == "<30").mean()), 3),
                     "reject_rate": round(float(b.rejected.mean()), 4),
                     "reject_rate_by_age": {g: round(float(v), 4) for g, v in rate["age_band"].items()},
                     "fairness_gap": round(gap, 4), "max_psi_feature": top, "max_psi": round(psis[top], 4),
                     "fairness_alert": gap > cfg["fair_limit"], "data_drift_alert": psis[top] > cfg["psi_limit"]})
        r = rows[-1]
        flag = ("RETRAIN TRIGGERED" if (r["fairness_alert"] or r["data_drift_alert"]) else "ok")
        print(f"batch {r['batch']}: under30={r['share_under_30']:.2f} reject={r['reject_rate']:.3f} "
              f"fairness gap={r['fairness_gap']:.3f} max PSI={r['max_psi']:.3f} ({top}) -> {flag}")
    first = next((r["batch"] for r in rows if r["fairness_alert"] or r["data_drift_alert"]), None)
    save_json({"limits": cfg, "batches": rows, "retrain_triggered_at_batch": first}, REPORTS / "monitoring.json")

    x = [r["batch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(x, [r["fairness_gap"] for r in rows], "o-", color="#8B1E2E", label="rejection-rate gap")
    ax[0].axhline(cfg["fair_limit"], ls="--", color="grey", label=f"limit {cfg['fair_limit']}")
    ax[0].set(title="Fairness drift (label-free)", xlabel="production batch", ylabel="max gap between groups"); ax[0].legend()
    ax[1].plot(x, [r["max_psi"] for r in rows], "o-", color="#1F3864", label="max PSI across features")
    ax[1].axhline(cfg["psi_limit"], ls="--", color="grey", label=f"limit {cfg['psi_limit']}")
    ax[1].set(title="Data drift (PSI)", xlabel="production batch", ylabel="PSI"); ax[1].legend()
    if first:
        for a in ax: a.axvline(first, color="#C9781E", alpha=0.6)
    fig.tight_layout(); fig.savefig(REPORTS / "monitoring.png", dpi=140)
    print(f"monitor: retrain trigger first fired at batch {first}" if first else "monitor: no drift alerts")

if __name__ == "__main__":
    main()

