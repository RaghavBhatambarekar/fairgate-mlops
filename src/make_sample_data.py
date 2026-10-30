"""Generate a SYNTHETIC file with the same columns as Kaggle's application_train.csv.

Used for CI and for a quick demo when the real data is not downloaded yet.
It deliberately contains age/gender related proxies so the fairness gate has
something real to catch. Results on this file are NOT results on Home Credit.
"""
import argparse
import numpy as np
import pandas as pd
from common import ROOT

def make(n=60000, seed=7):
    r = np.random.default_rng(seed)
    age = r.uniform(21, 69, n)
    male = r.random(n) < 0.34
    ext2 = np.clip(r.beta(4, 3, n) + 0.004 * (age - 45), 0, 1)
    ext3 = np.clip(r.beta(4, 3, n) + 0.006 * (age - 45), 0, 1)
    ext1 = np.clip(r.beta(3, 3, n) + 0.005 * (age - 45), 0, 1)
    income = np.exp(r.normal(11.9, 0.5, n)) * np.where(male, 1.15, 1.0)
    credit = income * r.uniform(1, 6, n)
    annuity = credit / r.uniform(10, 30, n)
    employed = -r.uniform(100, 9000, n)
    pension = age > 60
    employed[pension & (r.random(n) < 0.7)] = 365243
    logit = (-2.95 - 3.4 * (ext2 - 0.55) - 2.6 * (ext3 - 0.55) + 0.25 * (credit / income > 4)
             + 0.35 * male + 0.45 * (age < 30) - 0.25 * (age > 50) + r.normal(0, 0.4, n))
    target = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    ch = lambda opts, p=None: r.choice(opts, n, p=p)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100002, 100002 + n), "TARGET": target,
        "NAME_CONTRACT_TYPE": ch(["Cash loans", "Revolving loans"], [0.9, 0.1]),
        "CODE_GENDER": np.where(male, "M", "F"), "FLAG_OWN_CAR": ch(["Y", "N"], [0.34, 0.66]),
        "FLAG_OWN_REALTY": ch(["Y", "N"], [0.69, 0.31]), "CNT_CHILDREN": r.poisson(0.4, n),
        "AMT_INCOME_TOTAL": income.round(1), "AMT_CREDIT": credit.round(1), "AMT_ANNUITY": annuity.round(1),
        "AMT_GOODS_PRICE": (credit * 0.9).round(1),
        "NAME_INCOME_TYPE": np.where(pension, "Pensioner", ch(["Working", "Commercial associate", "State servant"], [0.65, 0.25, 0.1])),
        "NAME_EDUCATION_TYPE": ch(["Secondary / secondary special", "Higher education", "Incomplete higher"], [0.71, 0.24, 0.05]),
        "NAME_FAMILY_STATUS": ch(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], [0.64, 0.15, 0.1, 0.06, 0.05]),
        "NAME_HOUSING_TYPE": ch(["House / apartment", "With parents", "Municipal apartment", "Rented apartment"], [0.89, 0.05, 0.04, 0.02]),
        "REGION_POPULATION_RELATIVE": r.uniform(0.001, 0.07, n).round(6),
        "DAYS_BIRTH": (-age * 365.25).astype(int), "DAYS_EMPLOYED": employed.astype(int),
        "DAYS_REGISTRATION": -r.uniform(0, 15000, n).round(), "DAYS_ID_PUBLISH": -r.uniform(0, 6000, n).round(),
        "OWN_CAR_AGE": np.where(r.random(n) < 0.66, np.nan, r.integers(0, 30, n)),
        "OCCUPATION_TYPE": np.where(r.random(n) < 0.31, None, ch(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers"])),
        "CNT_FAM_MEMBERS": r.integers(1, 5, n),
        "EXT_SOURCE_1": np.where(r.random(n) < 0.56, np.nan, ext1), "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": np.where(r.random(n) < 0.2, np.nan, ext3),
    })
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60000); a = ap.parse_args()
    out = ROOT / "data" / "raw" / "application_train.csv"
    df = make(a.n); df.to_csv(out, index=False)
    print(f"synthetic sample written: {out} rows={len(df)} default_rate={df.TARGET.mean():.3f}")

