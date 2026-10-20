# Financial Risk Prediction

Credit-card default risk classifier (low-risk 0 / high-risk 1) built on the UCI Default of
Credit Card Clients dataset. Python training pipeline + FastAPI backend + HTML/JS frontend.
Educational demonstration; not financial advice or a lending decision.

## Run

```
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py                  # downloads data if data/ is empty, trains, writes artifacts/
uvicorn app:app --reload         # open http://127.0.0.1:8000  (API docs: /docs)
pytest -q                        # unit + API tests
```

If the automatic download is blocked, get the `.xls` from
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients and put it in `data/`.

Tabs can be opened directly: `/#predict`, `/#batch`, `/#report`, `/#about`.

## Files

| File | Purpose |
|---|---|
| `features.py` | Row-wise feature engineering shared by training and the API |
| `data_loader.py` | Download, load, schema checks, code audit |
| `train.py` | De-duplication, 70/15/15 stratified split, CV tuning, threshold + model selection on validation, one test evaluation with bootstrap CIs |
| `app.py` | `/api/health`, `/api/metrics`, `/api/predict`, `/api/predict_batch`, `/api/batch_template` and the static frontend |
| `static/index.html` | Risk form, batch CSV upload, model report, about & limitations |
| `artifacts/` | `metrics.json`, `manifest.json`, `splits.csv`, `model.joblib` |
| `Dockerfile` | Container for hosting the trained model |

## Design decisions

- The 35 exact duplicate rows are removed before splitting so no record can appear in both training and test.
- Demographics (SEX, EDUCATION, MARRIAGE, AGE) are excluded from model inputs and used only for the group error audit.
- Repayment codes −2 / −1 / 0 are counted separately and never treated as months of delay.
- Imputer and scaler live inside the Pipeline, so they are fitted on training folds only.
- Model chosen by validation PR-AUC among the **proposed** models (Logistic Regression, Random Forest);
  each threshold maximises validation F1; the test set is used once.
- Gradient Boosting is trained as an *extension* comparator. It is reported but not eligible for selection
  (`SELECT_FROM_EXTENSIONS = False` in `train.py`). Its validation PR-AUC was only 0.001 above Random
  Forest, well inside the noise shown by the confidence intervals.
- `requirements.txt` pins exact versions because `model.joblib` is tied to the scikit-learn version that saved it.

## Measured results (seed 42, scikit-learn 1.9.1)

Data: 30,000 rows → 29,965 after removing 35 duplicates; 0 missing cells; default rate 22.1%;
undocumented EDUCATION codes 0/5/6 and MARRIAGE code 0 merged into "other"; 3,932 negative bill cells.
Split: 20,975 / 4,495 / 4,495.

Validation (used for selection):

| Model | Threshold | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Baseline (majority) | 0.50 | 0.221 | 0.500 | 0.0% | 0.0% | 0.000 |
| Logistic Regression (C=0.1) | 0.30 | 0.538 | 0.757 | 54.5% | 52.1% | 0.532 |
| **Random Forest** (depth 10, leaf 5) ✔ | 0.31 | 0.564 | 0.777 | 54.3% | 54.7% | 0.545 |
| Gradient Boosting *(extension)* | 0.30 | 0.565 | 0.778 | 55.0% | 53.1% | 0.540 |

Final test (4,495 records, evaluated once) — Random Forest @ 0.31, with 95% bootstrap CIs (1,000 resamples):

| Metric | Random Forest | 95% CI | Baseline |
|---|---|---|---|
| Accuracy | 79.3% | 78.1 – 80.5% | 77.9% |
| Precision (high-risk) | 53.2% | 50.0 – 56.4% | 0% |
| Recall (high-risk) | 54.0% | 50.7 – 57.1% | 0% |
| F1 | 0.536 | 0.508 – 0.561 | 0 |
| ROC-AUC | 0.784 | 0.766 – 0.800 | 0.500 |
| PR-AUC | 0.563 | 0.531 – 0.595 | 0.221 |
| Brier score (lower is better) | 0.134 | – | 0.221 |

Confusion matrix (test): TN 3028, FP 473, FN 457, TP 537.

- **Importance:** `PAY_0_delay` (most recent month's delay) dominates, followed by `n_months_delayed` and `max_delay`.
- **Calibration:** predicted probabilities track observed default rates closely across all ten bins.
- **Accuracy is misleading here:** it barely moves from the baseline (77.9% → 79.3%) while recall goes from 0% to 54%.
- **Group audit:** recall ranges from about 50% (ages 30–39, graduate school) to 58% (under 30). The
  "Other/unknown" education group has only 64 records, so its figures are unreliable.

## Deploying a live link (optional)

The trained `artifacts/` folder is committed, so the container needs no training step.

- **Hugging Face Spaces:** create a Space with SDK "Docker", push this folder to it. It serves on port 7860.
- **Render:** New → Web Service → connect the GitHub repo → runtime "Docker". Render sets `$PORT` automatically.
- **Local Docker test:** `docker build -t risk . && docker run -p 7860:7860 risk`, then open http://127.0.0.1:7860
