"""Runs the whole FairGate pipeline end to end (same steps CI runs).

prepare -> train -> GATE(candidate) --pass--> register -> monitor -> report
                          |fail
                          v
                    mitigate -> GATE(mitigated) --pass--> register
                                     |fail -> pipeline stops, nothing is deployed
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"; REP = ROOT / "reports"

def step(name, *args, allow_fail=False):
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run([sys.executable, str(SRC / f"{name}.py"), *args], cwd=SRC)
    if r.returncode != 0 and not allow_fail:
        print(f"\nPIPELINE STOPPED at {name}"); sys.exit(r.returncode)
    return r.returncode == 0

def report():
    gates = {m: json.loads((REP / f"gate_{m}.json").read_text()) for m in ("candidate", "mitigated")
             if (REP / f"gate_{m}.json").exists()}
    mon = json.loads((REP / "monitoring.json").read_text())
    img = base64.b64encode((REP / "monitoring.png").read_bytes()).decode()
    def gate_html(m, g):
        badge = "<b style='color:#1a7f37'>PASSED</b>" if g["passed"] else "<b style='color:#c62828'>FAILED (blocked)</b>"
        rows = "".join(f"<tr><td>{col}</td><td>{grp}</td><td>{v['reject_rate']:.3f}</td><td>{v['tpr']:.3f}</td></tr>"
                       for col, d in g["groups"].items() for grp, v in d["by_group"].items())
        return (f"<h3>Model: {m} &nbsp; {badge}</h3><p>AUC {g['auc']:.4f} &middot; overall reject rate {g['reject_rate']:.3f} "
                f"&middot; demographic parity gap {g['worst_dp_diff']:.3f} &middot; equal opportunity gap {g['worst_eo_diff']:.3f}</p>"
                f"<table><tr><th>Attribute</th><th>Group</th><th>Reject rate</th><th>TPR (defaulters caught)</th></tr>{rows}</table>")
    mrows = "".join(f"<tr><td>{r['batch']}</td><td>{r['share_under_30']:.2f}</td><td>{r['reject_rate']:.3f}</td>"
                    f"<td>{r['fairness_gap']:.3f}</td><td>{r['max_psi']:.3f} ({r['max_psi_feature']})</td>"
                    f"<td>{'RETRAIN' if r['fairness_alert'] or r['data_drift_alert'] else 'ok'}</td></tr>" for r in mon["batches"])
    html = f"""<!doctype html><meta charset=utf-8><title>FairGate report</title>
<style>body{{font-family:Arial;max-width:1000px;margin:30px auto;color:#222}}h1{{color:#8B1E2E}}table{{border-collapse:collapse;margin:8px 0 18px}}
td,th{{border:1px solid #ccc;padding:5px 10px}}th{{background:#8B1E2E;color:#fff}}</style>
<h1>FairGate pipeline report</h1>
<h2>1. Fairness gate</h2>{''.join(gate_html(m, g) for m, g in gates.items())}
<h2>2. Production monitoring (simulated)</h2><img src="data:image/png;base64,{img}" width="100%">
<table><tr><th>Batch</th><th>Share under 30</th><th>Reject rate</th><th>Fairness gap</th><th>Max PSI</th><th>Status</th></tr>{mrows}</table>
<p>Retrain first triggered at batch: <b>{mon['retrain_triggered_at_batch']}</b></p>"""
    (REP / "fairgate_report.html").write_text(html)
    print(f"\nreport written: {REP / 'fairgate_report.html'}")

if __name__ == "__main__":
    step("prepare"); step("train")
    if step("gate", "--model", "candidate", allow_fail=True):
        step("register", "--model", "candidate")
    else:
        print("\n>>> candidate BLOCKED by fairness gate -> running mitigation")
        step("mitigate"); step("gate", "--model", "mitigated"); step("register", "--model", "mitigated")
    step("monitor"); report()

