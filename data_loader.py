"""Acquire, load, validate and clean the UCI Default of Credit Card Clients data."""
import datetime
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from features import PAY_COLS, RAW_INPUT_COLS, DEMOGRAPHIC_COLS

UCI_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATA_DIR = Path(__file__).parent / "data"
TARGET = "target"
REQUIRED = RAW_INPUT_COLS + DEMOGRAPHIC_COLS + [TARGET]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c.lower().replace(".", " ").replace("_", " ") == "default payment next month":
            rename[c] = TARGET
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        rename["PAY_1"] = "PAY_0"
    df = df.rename(columns=rename)
    if "ID" in df.columns:  # identifies a record, not credit behaviour
        df = df.drop(columns=["ID"])
    return df.apply(pd.to_numeric, errors="coerce")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return _normalise(pd.read_csv(path))
    df = pd.read_excel(path, header=1)  # UCI .xls has an X1..X23 row above real names
    if "LIMIT_BAL" not in [str(c).strip() for c in df.columns]:
        df = pd.read_excel(path, header=0)
    return _normalise(df)


def _download() -> Path:
    print(f"No dataset in {DATA_DIR}/ — downloading from UCI ...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=90) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx")))
        out = DATA_DIR / Path(name).name
        out.write_bytes(z.read(name))
    return out


def load_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in (".csv", ".xls", ".xlsx"))
    if not files:
        try:
            files = [_download()]
        except Exception as e:
            raise SystemExit(
                f"Automatic download failed ({e}).\nDownload the dataset manually from\n"
                f"  {UCI_PAGE}\nand place the .xls (or .csv) file inside {DATA_DIR}/"
            )
    path = files[0]
    df = _read(path)
    if TARGET not in df.columns:
        raise SystemExit(f"Could not find the target column in {path.name}.")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    source = {
        "file": path.name,
        "source_page": UCI_PAGE,
        "file_acquired": datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "citation": "Yeh, I.-C. Default of Credit Card Clients. UCI Machine Learning Repository.",
    }
    return df, source


def validate_dataset(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Dataset is missing columns: {missing_cols}")

    def codes(cols):
        vals = pd.unique(df[cols].values.ravel())
        return sorted(int(v) for v in vals if pd.notna(v))

    pay_codes = codes(PAY_COLS)
    counts = df[TARGET].value_counts().to_dict()
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df[REQUIRED].isna().sum().sum()),
        "duplicate_rows_excluding_id": int(df.duplicated().sum()),
        "class_counts": {"no_default_0": int(counts.get(0, 0)), "default_1": int(counts.get(1, 0))},
        "default_rate": round(float(df[TARGET].mean()), 4),
        "pay_status_codes_found": pay_codes,
        "unexpected_pay_codes": [v for v in pay_codes if v < -2 or v > 8],
        "education_codes_found": codes(["EDUCATION"]),
        "marriage_codes_found": codes(["MARRIAGE"]),
        "negative_bill_cells": int((df[[f"BILL_AMT{i}" for i in range(1, 7)]] < 0).sum().sum()),
    }
    return report


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Undocumented codes merged into 'other' (used only for the group-error audit)."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 3)
    return df
