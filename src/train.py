"""Train, select (CV on TRAIN only) and finally evaluate (TEST, once) the models.

Run from the project root:  python -m src.train
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data_utils import (DATA_PATH, FIG_DIR, MODEL_DIR, MODEL_PATH, OUTPUT_DIR, SEED,
                         load_data, make_split, split_features_target)
from .eda import run_eda
from .pipeline import build_models


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_training(data_path=DATA_PATH, cv_folds: int = 5, run_eda_step: bool = True) -> dict:
    for d in (OUTPUT_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    if run_eda_step:
        run_eda(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_split(X, y)
    classes = sorted(y.unique())
    print(f"Train: {X_train.shape}  Test (held out): {X_test.shape}  Features used: {X.shape[1]}")

    # ---- 1) MODEL SELECTION: cross-validation on the TRAINING set only -------------------
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    fitted, cv_scores, best_params = {}, {}, {}
    for name, (est, grid) in build_models().items():
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        fitted[name], cv_scores[name], best_params[name] = gs.best_estimator_, gs.best_score_, gs.best_params_
        print(f"[CV] {name:<36} macro-F1 = {gs.best_score_:.4f}  params = {gs.best_params_}")
    best_name = max(cv_scores, key=cv_scores.get)  # chosen WITHOUT looking at the test set
    print(f"\nSelected final model (highest CV macro-F1 on training data): {best_name}\n")

    # ---- 2) FINAL EVALUATION: held-out test set, each model evaluated once ---------------
    rows = []
    for name, model in fitted.items():
        pred = model.predict(X_test)
        rep = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
        row = {"model": name, "selected": name == best_name, "cv_macro_f1": cv_scores[name],
               "test_accuracy": rep["accuracy"],
               "test_macro_precision": rep["macro avg"]["precision"],
               "test_macro_recall": rep["macro avg"]["recall"],
               "test_macro_f1": rep["macro avg"]["f1-score"],
               "test_weighted_f1": rep["weighted avg"]["f1-score"]}
        for c in classes:
            for m in ("precision", "recall", "f1-score"):
                row[f"{c}_{m}"] = rep[c][m]
        rows.append(row)

        (OUTPUT_DIR / f"classification_report_{_safe(name)}.txt").write_text(
            f"{name}\nbest params: {best_params[name]}\n\n"
            + classification_report(y_test, pred, labels=classes, digits=3, zero_division=0),
            encoding="utf-8")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=classes, ax=axes[0],
                                                cmap="Blues", colorbar=False)
        axes[0].set_title("Counts")
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=classes, ax=axes[1],
                                                cmap="Blues", colorbar=False, normalize="true",
                                                values_format=".2f")
        axes[1].set_title("Row-normalised (= per-class recall)")
        fig.suptitle(f"Confusion matrix - {name} (test set)"); fig.tight_layout()
        fig.savefig(FIG_DIR / f"cm_{_safe(name)}.png", dpi=130); plt.close(fig)

    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(results.round(4).to_string(index=False))

    # ---- 3) Save final pipeline (preprocessing + model together) -------------------------
    final = fitted[best_name]
    joblib.dump(final, MODEL_PATH)

    # ---- 4) Interpretation: permutation importance of the final pipeline (test set) ------
    # Used for interpretation only - never for model selection.
    pi = permutation_importance(final, X_test, y_test, scoring="f1_macro",
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns, "importance_mean": pi.importances_mean,
                         "importance_std": pi.importances_std})
           .sort_values("importance_mean", ascending=False))
    imp.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3a6ea5")
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    ax.set_title(f"Permutation importance - {best_name}\n(association, NOT causation)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "permutation_importance.png", dpi=130); plt.close(fig)

    meta = {"selected_model": best_name, "selection_rule": "highest 5-fold CV macro-F1 on training set",
            "best_params": {k: str(v) for k, v in best_params.items()}, "seed": SEED,
            "classes": classes, "n_train": len(X_train), "n_test": len(X_test),
            "n_features": X.shape[1], "python": sys.version.split()[0],
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"results": results, "best_name": best_name, "importance": imp, "meta": meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA_PATH), help="path to semicolon-delimited data.csv")
    ap.add_argument("--cv-folds", type=int, default=5)
    a = ap.parse_args()
    run_training(a.data, a.cv_folds)
