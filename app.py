"""FastAPI backend. Run:  uvicorn app:app --reload  then open http://127.0.0.1:8000"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, create_model

from features import BILL_COLS, INPUT_GROUPS, PAY_COLS, PAYAMT_COLS, RAW_INPUT_COLS, engineer_features

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
DISCLAIMER = "Educational demonstration; not financial advice or a lending decision."
MAX_BATCH_ROWS = 20_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Financial Risk Prediction API", version="1.1")

# ---- Input validation (every user-supplied field is range-checked) ----
# (low, high, low_inclusive) — shared by the single and batch endpoints
RANGES = {"LIMIT_BAL": (0, 5_000_000, False)}
RANGES.update({c: (-2, 8, True) for c in PAY_COLS})
RANGES.update({c: (-1_000_000, 5_000_000, True) for c in BILL_COLS})
RANGES.update({c: (0, 5_000_000, True) for c in PAYAMT_COLS})


def _field(c, desc):
    lo, hi, inclusive = RANGES[c]
    return Field(..., le=hi, description=desc, **({"ge": lo} if inclusive else {"gt": lo}))


_fields = {"LIMIT_BAL": (float, _field("LIMIT_BAL", "Credit limit (NT$)"))}
_fields.update({c: (int, _field(c, "Repayment status code")) for c in PAY_COLS})
_fields.update({c: (float, _field(c, "Bill statement (NT$)")) for c in BILL_COLS})
_fields.update({c: (float, _field(c, "Amount paid (NT$)")) for c in PAYAMT_COLS})
Applicant = create_model("Applicant", **_fields)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        path = ART / "model.joblib"
        if not path.exists():
            raise HTTPException(503, "Model not trained yet. Run: python train.py")
        _bundle = joblib.load(path)
    return _bundle


def _proba(bundle, raw: pd.DataFrame):
    X = engineer_features(raw)[bundle["feature_names"]]
    return bundle["pipeline"].predict_proba(X)[:, 1]


def _explain(bundle, raw: pd.DataFrame):
    """What-if explanation: replace each input group with the training median and
    measure how the estimated probability changes. Associational, not causal."""
    original = raw.iloc[0].to_dict()
    rows = [original]
    for cols in INPUT_GROUPS.values():
        row = dict(original)
        for c in cols:
            row[c] = bundle["raw_medians"][c]
        rows.append(row)
    probs = _proba(bundle, pd.DataFrame(rows))
    drivers = [{"group": g, "effect": round(float(probs[0] - probs[i + 1]), 4)}
               for i, g in enumerate(INPUT_GROUPS)]
    drivers.sort(key=lambda d: -abs(d["effect"]))
    return float(probs[0]), drivers


@app.get("/api/health")
def health():
    try:
        b = get_bundle()
        return {"model_loaded": True, "model": b["model_name"], "threshold": b["threshold"]}
    except HTTPException:
        return {"model_loaded": False}


@app.get("/api/metrics")
def metrics():
    path = ART / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Metrics pending: run python train.py")
    return json.loads(path.read_text())


@app.post("/api/predict")
def predict(applicant: Applicant):
    # Nothing is stored: no borrower profile is saved (Stage Two §6.6)
    b = get_bundle()
    raw = pd.DataFrame([applicant.model_dump()])
    p, drivers = _explain(b, raw)
    f = engineer_features(raw).iloc[0]
    high = p >= b["threshold"]
    return {
        "probability": round(p, 4),
        "threshold": b["threshold"],
        "label": "High risk" if high else "Low risk",
        "class": int(high),
        "model": b["model_name"],
        "drivers": drivers,
        "indicators": {
            "months_delayed": int(f["n_months_delayed"]),
            "max_delay_months": int(f["max_delay"]),
            "utilisation_proxy": round(float(f["utilisation_proxy"]), 3),
            "recent_payment_coverage": round(float(f["coverage_recent"]), 3),
            "avg_bill": round(float(f["avg_bill"]), 0),
            "avg_payment": round(float(f["avg_payment"]), 0),
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/batch_template", response_class=PlainTextResponse)
def batch_template():
    steady = [200000, -1, -1, 0, 0, -1, -1, 15000, 18000, 22000, 20000, 16000, 14000,
              18000, 22000, 4000, 4000, 14000, 12000]
    stressed = [50000, 2, 2, 2, 0, 0, 0, 48500, 47800, 46200, 45000, 44100, 43000,
                0, 0, 1500, 1700, 1600, 1600]
    lines = [["ref"] + RAW_INPUT_COLS, ["A-001"] + steady, ["A-002"] + stressed]
    return "\n".join(",".join(map(str, l)) for l in lines) + "\n"


def _validate_batch(df: pd.DataFrame):
    missing = [c for c in RAW_INPUT_COLS if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing columns: {', '.join(missing)}")
    if len(df) == 0:
        raise HTTPException(422, "The file has no data rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(422, f"Too many rows ({len(df)}); the limit is {MAX_BATCH_ROWS}.")
    errors = []
    num = df[RAW_INPUT_COLS].apply(pd.to_numeric, errors="coerce")
    for c in RAW_INPUT_COLS:
        lo, hi, inclusive = RANGES[c]
        v = num[c]
        bad = v.isna() | (v > hi) | ((v < lo) if inclusive else (v <= lo))
        if c in PAY_COLS:
            bad |= v.notna() & (v != np.floor(v))
        for i in np.flatnonzero(bad.to_numpy())[:3]:
            errors.append(f"row {i + 2}, {c}: '{df[c].iloc[i]}' is not valid")
    if errors:
        raise HTTPException(422, "Invalid values — " + "; ".join(errors[:10]))
    return num


@app.post("/api/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # The uploaded file is processed in memory and never written to disk
    blob = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 5 MB.")
    try:
        df = pd.read_csv(io.BytesIO(blob))
    except Exception:
        raise HTTPException(422, "Could not read the file as CSV.")
    df.columns = [str(c).strip() for c in df.columns]
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})
    num = _validate_batch(df)
    b = get_bundle()
    p = _proba(b, num)
    ref = df["ref"].astype(str) if "ref" in df.columns else pd.Series(range(1, len(df) + 1)).astype(str)
    high = p >= b["threshold"]
    return {
        "model": b["model_name"],
        "threshold": b["threshold"],
        "rows": [{"ref": rf, "probability": round(float(pp), 4), "label": "High risk" if h else "Low risk"}
                 for rf, pp, h in zip(ref, p, high)],
        "summary": {"n": int(len(p)), "n_high": int(high.sum()), "mean_probability": round(float(p.mean()), 4)},
        "disclaimer": DISCLAIMER,
    }


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
