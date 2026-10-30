"""Shared config, feature definitions and fairness metrics used by every stage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for d in (ROOT / "data" / "raw", PROC, MODELS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET"
NUMERIC = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
           "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_POPULATION_RELATIVE",
           "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE"]
CATEGORICAL = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
               "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
ENGINEERED = ["credit_income_ratio", "annuity_income_ratio", "credit_term", "goods_credit_ratio"]
# Sensitive attributes are NOT model inputs; they are only used to measure (and mitigate) bias.
SENSITIVE_RAW = ["CODE_GENDER", "DAYS_BIRTH"]
FEATURES = NUMERIC + ENGINEERED + CATEGORICAL


def age_band(days_birth):
    age = -np.asarray(days_birth, dtype=float) / 365.25
    return np.where(age < 30, "<30", np.where(age <= 50, "30-50", ">50"))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)   # Home Credit placeholder value
    df["credit_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["credit_term"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["goods_credit_ratio"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["gender"] = df["CODE_GENDER"].astype(str)
    df["age_band"] = age_band(df["DAYS_BIRTH"])
    return df


def to_model_frame(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in NUMERIC + ENGINEERED:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype("object"), categories=categories[c])
    return X


def fairness_report(y_true, y_pred, y_score, sens: pd.DataFrame, gate_cfg: dict) -> dict:
    """y_pred = 1 means REJECTED (predicted default). Returns metrics + pass/fail."""
    out = {"auc": float(roc_auc_score(y_true, y_score)), "reject_rate": float(np.mean(y_pred)), "groups": {}}
    worst_dp, worst_eo = 0.0, 0.0
    for col in gate_cfg["sensitive"]:
        mf = MetricFrame(metrics={"reject_rate": selection_rate, "tpr": true_positive_rate},
                         y_true=y_true, y_pred=y_pred, sensitive_features=sens[col])
        g = mf.by_group
        dp = float(g["reject_rate"].max() - g["reject_rate"].min())
        eo = float(g["tpr"].max() - g["tpr"].min())
        out["groups"][col] = {"by_group": {k: {m: round(float(v), 4) for m, v in r.items()} for k, r in g.iterrows()},
                              "dp_diff": round(dp, 4), "eo_diff": round(eo, 4)}
        worst_dp, worst_eo = max(worst_dp, dp), max(worst_eo, eo)
    out["worst_dp_diff"], out["worst_eo_diff"] = round(worst_dp, 4), round(worst_eo, 4)
    checks = {"auc": out["auc"] >= gate_cfg["min_auc"],
              "demographic_parity": worst_dp <= gate_cfg["max_dp_diff"],
              "equal_opportunity": worst_eo <= gate_cfg["max_eo_diff"]}
    out["checks"], out["passed"] = checks, all(checks.values())
    return out


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def setup_mlflow():
    """Tracking DB and artifacts both live at the project root, whatever the working directory."""
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("fairgate") is None:
        mlflow.create_experiment("fairgate", artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("fairgate")

