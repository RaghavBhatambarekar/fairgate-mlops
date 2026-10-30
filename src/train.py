"""Stage 2: train LightGBM, pick the decision threshold, log everything to MLflow."""
import joblib
import numpy as np
import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from common import setup_mlflow, ROOT, PARAMS, PROC, MODELS, TARGET, CATEGORICAL, FEATURES, to_model_frame

setup_mlflow()

def main():
    tr = pd.read_parquet(PROC / "train.parquet"); te = pd.read_parquet(PROC / "test.parquet")
    cats = {c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL}
    Xtr, Xte = to_model_frame(tr, cats), to_model_frame(te, cats)
    p = PARAMS["train"]
    model = LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                           num_leaves=p["num_leaves"], subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=PARAMS["data"]["seed"], verbose=-1)
    with mlflow.start_run(run_name="train") as run:
        model.fit(Xtr, tr[TARGET])
        # Decision rule: reject the riskiest X% (threshold chosen on training scores).
        thr = float(np.quantile(model.predict_proba(Xtr)[:, 1], 1 - p["target_reject_rate"]))
        auc = roc_auc_score(te[TARGET], model.predict_proba(Xte)[:, 1])
        mlflow.log_params({**p, "n_features": len(FEATURES), "threshold": round(thr, 4)})
        mlflow.log_metric("test_auc", auc)
        mlflow.lightgbm.log_model(model, name="model")
        bundle = {"kind": "threshold", "model": model, "threshold": thr, "categories": cats,
                  "run_id": run.info.run_id, "version": "candidate"}
        joblib.dump(bundle, MODELS / "candidate.joblib")
        print(f"train: test AUC={auc:.4f} threshold={thr:.4f} run_id={run.info.run_id}")

if __name__ == "__main__":
    main()

