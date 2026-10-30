"""Stage 5: register ONLY a gate-passed model in the MLflow Model Registry and promote it."""
import argparse
import json
import shutil
import sys
import mlflow
from mlflow import MlflowClient
from common import ROOT, MODELS, REPORTS

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
NAME = "fairgate-credit"

def main(model_name):
    rep = json.loads((REPORTS / f"gate_{model_name}.json").read_text())
    if not rep["passed"]:
        print(f"register: REFUSED, {model_name} did not pass the fairness gate"); sys.exit(1)
    import joblib
    bundle = joblib.load(MODELS / f"{model_name}.joblib")
    mv = mlflow.register_model(f"runs:/{bundle['run_id']}/model", NAME)
    client = MlflowClient()
    client.set_registered_model_alias(NAME, "production", mv.version)
    for k, v in {"gate_passed": "true", "decision_rule": bundle["kind"],
                 "dp_gap": str(rep["worst_dp_diff"]), "eo_gap": str(rep["worst_eo_diff"])}.items():
        client.set_model_version_tag(NAME, mv.version, k, v)
    shutil.copy(MODELS / f"{model_name}.joblib", MODELS / "production.joblib")
    print(f"register: {NAME} v{mv.version} -> alias 'production' (decision rule: {bundle['kind']})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="candidate")
    main(ap.parse_args().model)

