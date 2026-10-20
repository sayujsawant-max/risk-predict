"""Row-wise feature engineering shared by training and the API.

Every feature uses only fields from the same customer row, so nothing here
learns from other records. Learned steps (imputation, scaling) live inside
the scikit-learn Pipeline and are fitted on training data only.
"""
import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_INPUT_COLS = ["LIMIT_BAL"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
DEMOGRAPHIC_COLS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

# Groups used for the "what-if" explanation in the API
INPUT_GROUPS = {
    "Repayment status (6 months)": PAY_COLS,
    "Credit limit": ["LIMIT_BAL"],
    "Bill statements": BILL_COLS,
    "Payments made": PAYAMT_COLS,
}


def _safe_ratio(num, den, fill, lo, hi):
    """num/den where den > 0, otherwise `fill`; clipped to [lo, hi]."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, fill, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return np.clip(out, lo, hi)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["LIMIT_BAL"] = df["LIMIT_BAL"].astype(float)

    # Repayment status codes: -2 no consumption, -1 paid in full,
    # 0 revolving credit (minimum paid), 1..8 months of delay.
    # Special codes are NOT treated as delay durations (Stage Two §2.3).
    pay = df[PAY_COLS].astype(float)
    delay = pay.clip(lower=0)
    for c in PAY_COLS:
        X[f"{c}_delay"] = delay[c]
    X["n_months_delayed"] = (pay >= 1).sum(axis=1)
    X["max_delay"] = delay.max(axis=1)
    X["n_months_paid_full"] = (pay == -1).sum(axis=1)
    X["n_months_no_use"] = (pay == -2).sum(axis=1)
    X["n_months_revolving"] = (pay == 0).sum(axis=1)

    bills = df[BILL_COLS].astype(float)
    pays = df[PAYAMT_COLS].astype(float)
    X["avg_bill"] = bills.mean(axis=1)
    X["avg_payment"] = pays.mean(axis=1)
    X["latest_bill"] = bills["BILL_AMT1"]
    X["latest_payment"] = pays["PAY_AMT1"]

    # Utilisation proxy: statement-based, not a true outstanding-balance ratio
    X["utilisation_proxy"] = _safe_ratio(bills["BILL_AMT1"], X["LIMIT_BAL"], 0.0, -1, 3)

    # Payment coverage: payment in month i is made against the previous
    # statement (BILL_AMT i+1). Zero/negative bill -> nothing owed -> coverage 1.
    cov = [
        _safe_ratio(pays[f"PAY_AMT{i}"], bills[f"BILL_AMT{i + 1}"], 1.0, 0, 5)
        for i in range(1, 6)
    ]
    X["coverage_recent"] = cov[0]
    X["coverage_avg"] = np.mean(cov, axis=0)

    # Payment trend / consistency
    X["payment_cv"] = pays.std(axis=1, ddof=0) / (pays.mean(axis=1) + 1.0)
    X["bill_trend"] = (bills["BILL_AMT1"] - bills["BILL_AMT6"]) / (X["LIMIT_BAL"] + 1.0)
    return X
