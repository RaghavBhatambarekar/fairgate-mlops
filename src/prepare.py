"""Stage 1: validate raw data, engineer features, split train/test."""
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from common import ROOT, PARAMS, PROC, REPORTS, TARGET, NUMERIC, CATEGORICAL, SENSITIVE_RAW, engineer, save_json

def validate(df):
    """Lightweight data checks (the role Great Expectations plays in the full design)."""
    errors = []
    needed = [TARGET] + NUMERIC + CATEGORICAL + SENSITIVE_RAW
    missing = [c for c in needed if c not in df.columns]
    if missing: errors.append(f"missing columns: {missing}")
    else:
        if not set(df[TARGET].unique()) <= {0, 1}: errors.append("TARGET must be 0/1")
        if (df["AMT_INCOME_TOTAL"] <= 0).any(): errors.append("AMT_INCOME_TOTAL must be > 0")
        if (df["AMT_CREDIT"] <= 0).any(): errors.append("AMT_CREDIT must be > 0")
        if (df["DAYS_BIRTH"] >= 0).any(): errors.append("DAYS_BIRTH must be negative")
        if df["EXT_SOURCE_2"].isna().mean() > 0.05: errors.append("EXT_SOURCE_2 null rate above 5%")
    return errors

def main():
    raw = ROOT / PARAMS["data"]["raw_path"]
    df = pd.read_csv(raw)
    df = df[df["CODE_GENDER"].isin(["M", "F"])]           # drops the 4 'XNA' rows in Home Credit
    errors = validate(df)
    save_json({"rows": len(df), "errors": errors}, REPORTS / "data_validation.json")
    if errors:
        print("DATA VALIDATION FAILED:", *errors, sep="\n  "); sys.exit(1)
    df = engineer(df)
    tr, te = train_test_split(df, test_size=PARAMS["data"]["test_size"], stratify=df[TARGET],
                              random_state=PARAMS["data"]["seed"])
    tr.to_parquet(PROC / "train.parquet"); te.to_parquet(PROC / "test.parquet")
    print(f"prepare: rows={len(df)} train={len(tr)} test={len(te)} default_rate={df[TARGET].mean():.3f}")

if __name__ == "__main__":
    main()

