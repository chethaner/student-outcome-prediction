"""Demonstrate predictions with the SAVED pipeline (same preprocessing as training).

  python -m src.predict                 # 2 random held-out test students per class
  python -m src.predict --csv my.csv    # your own records (semicolon-delimited, same columns)
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .data_utils import (DATA_PATH, MODEL_PATH, OUTPUT_DIR, SEED, TARGET, load_data, make_split,
                         select_features, split_features_target)

SHOW = ["Course", "Age at enrollment", "Admission grade", "Gender", "Scholarship holder", "Debtor"]


def predict_frame(pipe, X: pd.DataFrame) -> pd.DataFrame:
    proba = pipe.predict_proba(X)
    out = pd.DataFrame(proba, columns=[f"P({c})" for c in pipe.classes_], index=X.index).round(3)
    out.insert(0, "predicted", pipe.predict(X))
    return out


def demo_predictions(n_per_class: int = 2) -> pd.DataFrame:
    pipe = joblib.load(MODEL_PATH)
    X, y = split_features_target(load_data(DATA_PATH))
    _, X_test, _, y_test = make_split(X, y)  # same seed -> same held-out students
    idx = pd.concat([y_test[y_test == c].sample(n_per_class, random_state=SEED) for c in pipe.classes_]).index
    res = predict_frame(pipe, X_test.loc[idx])
    res.insert(0, "actual", y_test.loc[idx])
    res["correct"] = res["actual"] == res["predicted"]
    return pd.concat([X_test.loc[idx, SHOW], res], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--n-per-class", type=int, default=2)
    a = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 30)
    if a.csv:
        df = load_data(a.csv)
        res = predict_frame(joblib.load(MODEL_PATH), select_features(df))
        if TARGET in df:
            res.insert(0, "actual", df[TARGET])
    else:
        res = demo_predictions(a.n_per_class)
    print(res)
    OUTPUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUTPUT_DIR / "sample_predictions.csv")
    print(f"\nSaved to {OUTPUT_DIR / 'sample_predictions.csv'}")
