"""Stage 4 (runs only when the gate fails): post-process the model with fairlearn ThresholdOptimizer.

It learns a separate decision threshold per group (gender x age band) so rejection rates
are equalised (demographic parity) while maximising balanced accuracy. The underlying
LightGBM model and its risk scores are unchanged; only the decision rule changes.
"""
import warnings
import joblib
import pandas as pd
import mlflow
from fairlearn.postprocessing import ThresholdOptimizer
from common import setup_mlflow, ROOT, PARAMS, PROC, MODELS, TARGET, to_model_frame
from predictor import group_key

warnings.filterwarnings("ignore")
setup_mlflow()

def main():
    b = joblib.load(MODELS / "candidate.joblib")
    tr = pd.read_parquet(PROC / "train.parquet")
    X = to_model_frame(tr, b["categories"])
    opt = ThresholdOptimizer(estimator=b["model"], constraints=PARAMS["gate"]["mitigation_constraint"],
                             objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    opt.fit(X, tr[TARGET], sensitive_features=group_key(tr))
    mitigated = {**b, "kind": "threshold_optimizer", "optimizer": opt, "version": "mitigated"}
    joblib.dump(mitigated, MODELS / "mitigated.joblib")
    with mlflow.start_run(run_name="mitigate"):
        mlflow.log_params({"method": "ThresholdOptimizer", "constraint": PARAMS["gate"]["mitigation_constraint"],
                           "objective": "balanced_accuracy_score", "groups": "gender x age_band",
                           "base_run_id": b["run_id"]})
    print("mitigate: ThresholdOptimizer fitted on gender x age_band groups -> models/mitigated.joblib")

if __name__ == "__main__":
    main()

