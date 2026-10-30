"""Run after the pipeline: pytest -q"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from common import fairness_report  # noqa: E402
from app.main import app  # noqa: E402

GATE = {"sensitive": ["g"], "max_dp_diff": 0.05, "max_eo_diff": 0.10, "min_auc": 0.5}

def test_gate_blocks_biased_decisions():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0]); score = np.linspace(0, 1, 8)
    sens = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4})
    biased = np.array([1, 1, 1, 1, 0, 0, 0, 0])          # rejects everyone in group a, nobody in b
    assert not fairness_report(y, biased, score, sens, GATE)["passed"]

def test_gate_passes_equal_decisions():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0]); score = y * 0.9 + 0.05
    sens = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4})
    assert fairness_report(y, y.copy(), score, sens, GATE)["passed"]

def test_api_returns_decision_and_three_reason_codes():
    c = TestClient(app)
    r = c.post("/predict", json={"CODE_GENDER": "F", "DAYS_BIRTH": -9500, "AMT_INCOME_TOTAL": 150000,
                                 "AMT_CREDIT": 700000, "AMT_ANNUITY": 32000, "AMT_GOODS_PRICE": 630000,
                                 "DAYS_EMPLOYED": -800, "EXT_SOURCE_2": 0.3, "EXT_SOURCE_3": 0.25})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in ("APPROVE", "REJECT")
    assert len(body["reason_codes"]) == 3

