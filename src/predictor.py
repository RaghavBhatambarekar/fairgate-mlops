"""One place that turns a model bundle + applicant rows into scores, decisions and reason codes."""
import numpy as np
import pandas as pd
from common import FEATURES, to_model_frame

def group_key(df: pd.DataFrame) -> pd.Series:
    return df["gender"].astype(str) + "|" + df["age_band"].astype(str)

def predict(bundle: dict, df: pd.DataFrame, seed: int = 0):
    """Returns (risk score, decision) where decision 1 = REJECT."""
    X = to_model_frame(df, bundle["categories"])
    score = bundle["model"].predict_proba(X)[:, 1]
    if bundle["kind"] == "threshold":
        decision = (score >= bundle["threshold"]).astype(int)
    else:  # fairlearn ThresholdOptimizer: group-specific thresholds learned during mitigation
        decision = bundle["optimizer"].predict(X, sensitive_features=group_key(df), random_state=seed).astype(int)
    return score, decision

def reason_codes(bundle: dict, df: pd.DataFrame, top_k: int = 3):
    """TreeSHAP contributions computed natively by LightGBM (pred_contrib=True)."""
    X = to_model_frame(df, bundle["categories"])
    contrib = bundle["model"].booster_.predict(X, pred_contrib=True)[:, :-1]  # last column = base value
    out = []
    for row, vals in zip(X.itertuples(index=False), contrib):
        idx = np.argsort(-np.abs(vals))[:top_k]   # strongest drivers, either direction
        out.append([{"feature": FEATURES[i], "value": None if pd.isna(row[i]) else (row[i] if isinstance(row[i], str) else float(row[i])),
                     "impact": round(float(vals[i]), 4)} for i in idx])
    return out

