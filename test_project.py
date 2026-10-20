"""Run:  pytest -q   (API tests are skipped until python train.py has been run)"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import RAW_INPUT_COLS, engineer_features

SAMPLE = {"LIMIT_BAL": 50000, "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": -1, "PAY_5": -2, "PAY_6": 0,
          "BILL_AMT1": 48000, "BILL_AMT2": 0, "BILL_AMT3": -500, "BILL_AMT4": 1000, "BILL_AMT5": 0, "BILL_AMT6": 900,
          "PAY_AMT1": 0, "PAY_AMT2": 100, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 50, "PAY_AMT6": 0}


def test_features_finite_with_zero_and_negative_bills():
    X = engineer_features(pd.DataFrame([SAMPLE]))
    assert np.isfinite(X.to_numpy()).all()


def test_special_pay_codes_not_counted_as_delay():
    f = engineer_features(pd.DataFrame([SAMPLE])).iloc[0]
    assert f["n_months_delayed"] == 2
    assert f["max_delay"] == 2
    assert f["n_months_paid_full"] == 1 and f["n_months_no_use"] == 1 and f["n_months_revolving"] == 2
    assert f["PAY_4_delay"] == 0 and f["PAY_5_delay"] == 0


def test_coverage_is_one_when_nothing_owed():
    f = engineer_features(pd.DataFrame([SAMPLE])).iloc[0]
    # PAY_AMT1 is paid against BILL_AMT2 == 0 -> coverage defaults to 1
    assert f["coverage_recent"] == 1.0


def test_features_are_row_independent():
    one = engineer_features(pd.DataFrame([SAMPLE]))
    other = dict(SAMPLE, LIMIT_BAL=999999, PAY_0=8)
    two = engineer_features(pd.DataFrame([SAMPLE, other]))
    pd.testing.assert_frame_equal(one, two.iloc[[0]])


trained = (Path(__file__).parent / "artifacts" / "model.joblib").exists()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


@pytest.mark.skipif(not trained, reason="run python train.py first")
def test_predict_returns_probability(client):
    r = client.post("/api/predict", json=SAMPLE)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probability"] <= 1
    assert j["label"] in ("High risk", "Low risk")
    assert len(j["drivers"]) == 4


@pytest.mark.skipif(not trained, reason="run python train.py first")
@pytest.mark.parametrize("field,value", [("PAY_0", 9), ("PAY_0", -3), ("LIMIT_BAL", 0), ("PAY_AMT1", -5)])
def test_predict_rejects_out_of_range(client, field, value):
    r = client.post("/api/predict", json=dict(SAMPLE, **{field: value}))
    assert r.status_code == 422


@pytest.mark.skipif(not trained, reason="run python train.py first")
def test_predict_rejects_missing_field(client):
    body = {k: v for k, v in SAMPLE.items() if k != "BILL_AMT3"}
    assert client.post("/api/predict", json=body).status_code == 422


@pytest.mark.skipif(not trained, reason="run python train.py first")
def test_metrics_and_health(client):
    assert client.get("/api/health").json()["model_loaded"] is True
    m = client.get("/api/metrics").json()
    assert m["test"]["baseline"]["recall"] == 0
    assert set(RAW_INPUT_COLS).isdisjoint(m["model_inputs_exclude"])
    assert m["data_report"]["duplicates_removed"] >= 0
    assert m["lr_coefficients"], "LR coefficients must be reported whichever model is selected"
    lo, hi = m["test"]["selected_model_ci95"]["recall"]
    assert lo <= m["test"]["selected_model"]["recall"] <= hi
    if not m["select_from_extensions"]:
        assert not m["models"][m["selected_model"]]["extension"]


@pytest.mark.skipif(not trained, reason="run python train.py first")
def test_batch_template_round_trip(client):
    csv = client.get("/api/batch_template").text
    r = client.post("/api/predict_batch", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200
    j = r.json()
    assert j["summary"]["n"] == 2 and [x["ref"] for x in j["rows"]] == ["A-001", "A-002"]
    # batch and single endpoints must agree
    single = client.post("/api/predict", json=SAMPLE).json()["probability"]
    one = pd.DataFrame([SAMPLE]).to_csv(index=False)
    batch = client.post("/api/predict_batch", files={"file": ("s.csv", one, "text/csv")}).json()
    assert batch["rows"][0]["probability"] == single


@pytest.mark.skipif(not trained, reason="run python train.py first")
@pytest.mark.parametrize("csv,msg", [
    ("a,b\n1,2\n", "Missing columns"),
    (",".join(RAW_INPUT_COLS) + "\n", "no data rows"),
    (",".join(RAW_INPUT_COLS) + "\n" + ",".join(["9"] * len(RAW_INPUT_COLS)) + "\n", "PAY_0"),
    (",".join(RAW_INPUT_COLS) + "\n" + ",".join(["x"] * len(RAW_INPUT_COLS)) + "\n", "not valid"),
])
def test_batch_rejects_bad_files(client, csv, msg):
    r = client.post("/api/predict_batch", files={"file": ("bad.csv", csv, "text/csv")})
    assert r.status_code == 422 and msg in r.json()["detail"]
