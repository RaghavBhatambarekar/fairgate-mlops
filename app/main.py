"""FairGate scoring API: risk score + approve/reject decision + top 3 SHAP reason codes."""
import sys
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import json
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import engineer  # noqa: E402
from predictor import predict, reason_codes  # noqa: E402

app = FastAPI(title="FairGate credit scoring API", version="1.0")
BUNDLE = None

def bundle():
    global BUNDLE
    if BUNDLE is None:
        path = ROOT / "models" / "production.joblib"
        if not path.exists():
            raise HTTPException(503, "No production model. Run the pipeline first (python run_pipeline.py).")
        BUNDLE = joblib.load(path)
    return BUNDLE

class Applicant(BaseModel):
    CODE_GENDER: str
    DAYS_BIRTH: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_EMPLOYED: int
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    REGION_POPULATION_RELATIVE: float = 0.02
    CNT_CHILDREN: int = 0
    CNT_FAM_MEMBERS: float = 2
    DAYS_REGISTRATION: float = -4000
    DAYS_ID_PUBLISH: float = -3000
    OWN_CAR_AGE: Optional[float] = None
    NAME_CONTRACT_TYPE: str = "Cash loans"
    NAME_INCOME_TYPE: str = "Working"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    OCCUPATION_TYPE: Optional[str] = None
    FLAG_OWN_CAR: str = "N"
    FLAG_OWN_REALTY: str = "Y"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (ROOT / "models" / "production.joblib").exists()}

@app.get("/model")
def model_info():
    b = bundle()
    return {"version": b["version"], "decision_rule": b["kind"], "mlflow_run_id": b["run_id"]}

@app.post("/predict")
def score(a: Applicant):
    b = bundle()
    df = engineer(pd.DataFrame([a.model_dump()]))
    risk, decision = predict(b, df)
    return {"default_risk": round(float(risk[0]), 4),
            "decision": "REJECT" if decision[0] == 1 else "APPROVE",
            "reason_codes": reason_codes(b, df)[0],
            "model_version": b["version"], "decision_rule": b["kind"]}


REPORTS = ROOT / "reports"

@app.get("/gate")
def gate_results():
    """Gate reports for every model that went through the fairness gate."""
    out = {}
    for m in ("candidate", "mitigated"):
        f = REPORTS / f"gate_{m}.json"
        if f.exists():
            g = json.loads(f.read_text())
            out[m] = {k: g[k] for k in ("passed", "auc", "reject_rate", "worst_dp_diff", "worst_eo_diff", "groups")}
    return out

@app.get("/monitoring.png")
def monitoring_chart():
    f = REPORTS / "monitoring.png"
    if not f.exists():
        raise HTTPException(404, "Run the pipeline to create the monitoring chart.")
    return FileResponse(f)

@app.get("/report", response_class=HTMLResponse)
def full_report():
    f = REPORTS / "fairgate_report.html"
    return f.read_text() if f.exists() else "<p>Run the pipeline to create the report.</p>"

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "app" / "index.html").read_text()

