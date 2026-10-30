"""Stage 3: the FAIRNESS GATE. Exit code 0 = pass, 1 = blocked. CI uses the exit code."""
import argparse
import sys
import joblib
import pandas as pd
import mlflow
from common import setup_mlflow, ROOT, PARAMS, PROC, MODELS, REPORTS, TARGET, fairness_report, save_json
from predictor import predict

setup_mlflow()

def run_gate(model_name: str) -> dict:
    bundle = joblib.load(MODELS / f"{model_name}.joblib")
    te = pd.read_parquet(PROC / "test.parquet")
    score, decision = predict(bundle, te)
    rep = fairness_report(te[TARGET].values, decision, score, te, PARAMS["gate"])
    rep["model"] = model_name
    save_json(rep, REPORTS / f"gate_{model_name}.json")
    with mlflow.start_run(run_name=f"gate_{model_name}"):
        mlflow.log_metrics({"auc": rep["auc"], "worst_dp_diff": rep["worst_dp_diff"],
                            "worst_eo_diff": rep["worst_eo_diff"], "gate_passed": int(rep["passed"])})
        mlflow.log_artifact(str(REPORTS / f"gate_{model_name}.json"))
    status = "PASSED" if rep["passed"] else "FAILED"
    print(f"gate[{model_name}] {status}: AUC={rep['auc']:.4f} reject_rate={rep['reject_rate']:.3f} "
          f"DP gap={rep['worst_dp_diff']:.4f} (limit {PARAMS['gate']['max_dp_diff']}) "
          f"EO gap={rep['worst_eo_diff']:.4f} (limit {PARAMS['gate']['max_eo_diff']})")
    for col, g in rep["groups"].items():
        rates = ", ".join(f"{k}: {v['reject_rate']:.3f}" for k, v in g["by_group"].items())
        print(f"    reject rate by {col}: {rates}")
    return rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="candidate")
    rep = run_gate(ap.parse_args().model)
    sys.exit(0 if rep["passed"] else 1)

