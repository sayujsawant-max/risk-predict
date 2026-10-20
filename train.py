"""Stage Three implementation: train, select, threshold, evaluate once, save artifacts.

Run:  python train.py
"""
import datetime
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import TARGET, clean_demographics, load_dataset, validate_dataset
from features import DEMOGRAPHIC_COLS, RAW_INPUT_COLS, engineer_features

SEED = 42
THRESHOLD_STRATEGY = "max_f1"   # or "recall_target"
RECALL_TARGET = 0.60            # used only when THRESHOLD_STRATEGY == "recall_target"
N_BOOTSTRAP = 1000
BASELINE = "Baseline (majority class)"
EXTENSION_MODELS = {"Gradient Boosting"}  # beyond the Stage One/Two proposal
SELECT_FROM_EXTENSIONS = False  # False: extensions are reported but only proposed models can be selected
ART = Path(__file__).parent / "artifacts"
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
GRID = np.linspace(0, 1, 101)


def r(x):
    return None if x is None else round(float(x), 4)


def make_pipe(estimator):
    # Imputer + scaler are fitted inside the pipeline -> training data only
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", estimator)])


def candidates():
    return {
        BASELINE: (make_pipe(DummyClassifier(strategy="most_frequent")), None),
        "Logistic Regression": (
            make_pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            make_pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [5, 20],
             "model__class_weight": [None, "balanced_subsample"]},
        ),
        "Gradient Boosting": (
            make_pipe(HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                                     random_state=SEED)),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
             "model__min_samples_leaf": [20, 80]},
        ),
    }


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": r(thr),
        "accuracy": r(accuracy_score(y, pred)),
        "precision": r(precision_score(y, pred, zero_division=0)),
        "recall": r(recall_score(y, pred, zero_division=0)),
        "f1": r(f1_score(y, pred, zero_division=0)),
        "roc_auc": r(roc_auc_score(y, proba)),
        "pr_auc": r(average_precision_score(y, proba)),
        "brier": r(brier_score_loss(y, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_curve(y, proba):
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"t": float(t),
                     "precision": r(precision_score(y, pred, zero_division=0)),
                     "recall": r(recall_score(y, pred, zero_division=0)),
                     "f1": r(f1_score(y, pred, zero_division=0)),
                     "accuracy": r((tp + tn) / len(y)),
                     "flagged_rate": r(pred.mean()),
                     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return rows


def choose_threshold(curve):
    if THRESHOLD_STRATEGY == "recall_target":
        ok = [c for c in curve if c["recall"] >= RECALL_TARGET]
        if ok:  # highest cutoff that still meets the recall target
            return max(ok, key=lambda c: c["t"])["t"]
    return max(curve, key=lambda c: c["f1"])["t"]


def curves(y, proba):
    """ROC and PR curves resampled onto a fixed 101-point grid for plotting."""
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    # Interpolated precision: best precision achievable at recall >= r
    pr = [float(prec[rec >= g].max()) if (rec >= g).any() else 0.0 for g in GRID]
    return {"roc": {"fpr": [r(v) for v in GRID], "tpr": [r(v) for v in np.interp(GRID, fpr, tpr)]},
            "pr": {"recall": [r(v) for v in GRID], "precision": [r(v) for v in pr]}}


def bootstrap_ci(y, proba, thr, n=N_BOOTSTRAP):
    """95% percentile intervals from resampling the test set with replacement."""
    rng = np.random.default_rng(SEED)
    pred = (proba >= thr).astype(int)
    stats = {k: [] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yt, yp, pp = y[i], pred[i], proba[i]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        stats["accuracy"].append((yt == yp).mean())
        stats["precision"].append(prec)
        stats["recall"].append(rec)
        stats["f1"].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        stats["roc_auc"].append(roc_auc_score(yt, pp))
        stats["pr_auc"].append(average_precision_score(yt, pp))
    return {k: [r(np.percentile(v, 2.5)), r(np.percentile(v, 97.5))] for k, v in stats.items()}


def group_errors(demo, y, pred):
    groupings = {
        "Sex": demo["SEX"].map({1: "Male", 2: "Female"}),
        "Education": demo["EDUCATION"].map({1: "Graduate school", 2: "University",
                                            3: "High school", 4: "Other/unknown"}),
        "Age band": pd.cut(demo["AGE"], [0, 29, 39, 49, 200],
                           labels=["<30", "30-39", "40-49", "50+"]).astype(str),
    }
    out = []
    for attr, g in groupings.items():
        g = g.to_numpy()
        for level in sorted(set(x for x in g if isinstance(x, str) and x != "nan")):
            m = g == level
            yt, yp = y[m], pred[m]
            pos, neg = int((yt == 1).sum()), int((yt == 0).sum())
            out.append({
                "attribute": attr, "group": level, "n": int(m.sum()),
                "default_rate": r(yt.mean()), "flagged_rate": r(yp.mean()),
                "recall": r(((yp == 1) & (yt == 1)).sum() / pos) if pos else None,
                "false_positive_rate": r(((yp == 1) & (yt == 0)).sum() / neg) if neg else None,
                "small_sample": bool(m.sum() < 100),
            })
    return out


def main():
    ART.mkdir(exist_ok=True)
    df, source = load_dataset()
    data_report = validate_dataset(df)
    # Identical rows could fall on both sides of the split and leak; keep one copy
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    data_report["rows_after_dedup"] = int(len(df))
    data_report["duplicates_removed"] = int(n_before - len(df))
    print("Data report:", json.dumps(data_report, indent=2))
    df = clean_demographics(df)

    y = df[TARGET].astype(int).to_numpy()
    X = engineer_features(df)          # demographics are NOT model inputs
    demo = df[DEMOGRAPHIC_COLS]        # kept only for the fairness audit

    idx = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED)
    Xtr, Xva, Xte = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    ytr, yva, yte = y[idx_train], y[idx_val], y[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, (pipe, grid) in candidates().items():
        print(f"Training {name} ...", flush=True)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
            gs.fit(Xtr, ytr)
            model = gs.best_estimator_
            best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
            cv_score = gs.best_score_
        else:
            model, best_params, cv_score = pipe.fit(Xtr, ytr), {}, None

        p_val = model.predict_proba(Xva)[:, 1]
        curve = threshold_curve(yva, p_val)
        thr = choose_threshold(curve) if name != BASELINE else 0.5
        results[name] = {
            "extension": name in EXTENSION_MODELS,
            "best_params": best_params,
            "cv_pr_auc_train": r(cv_score),
            "selected_threshold": thr,
            "validation_at_0.5": metrics_at(yva, p_val, 0.5),
            "validation_at_selected": metrics_at(yva, p_val, thr),
            "threshold_curve": curve,
            "validation_curves": curves(yva, p_val),
        }
        fitted[name] = model

    # Model selection on VALIDATION only
    contenders = [n for n in results if n != BASELINE
                  and (SELECT_FROM_EXTENSIONS or n not in EXTENSION_MODELS)]
    best_name = max(contenders, key=lambda n: results[n]["validation_at_selected"]["pr_auc"])
    best, thr = fitted[best_name], results[best_name]["selected_threshold"]
    print(f"Selected: {best_name} at threshold {thr}")

    # ONE final evaluation on the untouched test set
    p_test = best.predict_proba(Xte)[:, 1]
    pred_test = (p_test >= thr).astype(int)
    test_selected = metrics_at(yte, p_test, thr)
    test_baseline = metrics_at(yte, fitted[BASELINE].predict_proba(Xte)[:, 1], 0.5)
    print(f"Bootstrapping {N_BOOTSTRAP} test resamples ...", flush=True)
    test_ci = bootstrap_ci(yte, p_test, thr)
    frac_pos, mean_pred = calibration_curve(yte, p_test, n_bins=10, strategy="quantile")

    # Interpretation (validation data, not test)
    perm = permutation_importance(best, Xva, yva, scoring="average_precision",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = sorted(
        [{"feature": f, "mean": r(m), "std": r(s)}
         for f, m, s in zip(X.columns, perm.importances_mean, perm.importances_std)],
        key=lambda d: -d["mean"])
    # Standardised LR coefficients are reported whichever model is selected (Stage Two §6.2)
    coefs = fitted["Logistic Regression"].named_steps["model"].coef_[0]
    coefficients = sorted([{"feature": f, "coef": r(c)} for f, c in zip(X.columns, coefs)],
                          key=lambda d: -abs(d["coef"]))

    split_sizes = {s: {"n": int(len(i)), "default_rate": r(y[i].mean())}
                   for s, i in [("train", idx_train), ("validation", idx_val), ("test", idx_test)]}

    metrics = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_report": data_report,
        "splits": split_sizes,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "selected_model": best_name,
        "selected_threshold": thr,
        "select_from_extensions": SELECT_FROM_EXTENSIONS,
        "models": results,
        "test": {"selected_model": test_selected, "baseline": test_baseline,
                 "selected_model_ci95": test_ci, "n_bootstrap": N_BOOTSTRAP,
                 "curves": curves(yte, p_test)},
        "calibration_test": {"mean_predicted": [r(v) for v in mean_pred],
                             "observed_default_rate": [r(v) for v in frac_pos]},
        "permutation_importance_validation": importance,
        "lr_coefficients": coefficients,
        "group_errors_test": group_errors(demo.iloc[idx_test], yte, pred_test),
        "model_inputs_exclude": DEMOGRAPHIC_COLS,
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump({
        "pipeline": best, "model_name": best_name, "threshold": thr,
        "feature_names": list(X.columns),
        "raw_medians": df.iloc[idx_train][RAW_INPUT_COLS].median().to_dict(),
        "sklearn_version": sklearn.__version__,
    }, ART / "model.joblib", compress=3)

    pd.DataFrame({"row_index": np.concatenate([idx_train, idx_val, idx_test]),
                  "split": ["train"] * len(idx_train) + ["validation"] * len(idx_val)
                           + ["test"] * len(idx_test)}).to_csv(ART / "splits.csv", index=False)

    (ART / "manifest.json").write_text(json.dumps({
        "source": source, "random_seed": SEED, "split": "stratified 70/15/15 after de-duplication",
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "threshold_strategy": THRESHOLD_STRATEGY, "selected_model": best_name,
        "selected_threshold": thr,
    }, indent=2))
    print("Test metrics (selected model):", json.dumps(test_selected, indent=2))
    print("95% bootstrap CI:", json.dumps(test_ci))
    print(f"Artifacts written to {ART}/")


if __name__ == "__main__":
    main()
