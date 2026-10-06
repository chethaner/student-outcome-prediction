"""Sensitivity check: do 'Tuition fees up to date' / 'Debtor' (possibly recorded AFTER enrollment) drive the result?

Diagnostic only - NOT used for model selection. Same split, same seed, same model configurations as the
selected pipeline family; the test set is scored once per variant.   Run:  python -m src.sensitivity
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from .data_utils import OUTPUT_DIR, SEED, load_data, make_split, split_features_target
from .pipeline import build_preprocessor

VARIANTS = {
    "all 24 permitted features (primary)": (),
    "without 'Tuition fees up to date'": ("Tuition fees up to date",),
    "without 'Tuition fees up to date' and 'Debtor'": ("Tuition fees up to date", "Debtor"),
}


def main():
    X, y = split_features_target(load_data())
    X_train, X_test, y_train, y_test = make_split(X, y)
    # RF/LR settings = the best CV settings found by src.train for each family
    models = {
        "Random Forest (balanced)": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "Logistic Regression (balanced)": lambda: LogisticRegression(
            max_iter=3000, class_weight="balanced", C=0.1, random_state=SEED),
    }
    rows = []
    for vname, excl in VARIANTS.items():
        for mname, mk in models.items():
            pipe = Pipeline([("prep", build_preprocessor(excl)), ("clf", mk())]).fit(X_train, y_train)
            rep = classification_report(y_test, pipe.predict(X_test), output_dict=True, zero_division=0)
            rows.append({"features": vname, "model": mname, "test_accuracy": rep["accuracy"],
                         "test_macro_f1": rep["macro avg"]["f1-score"],
                         "Dropout_recall": rep["Dropout"]["recall"], "Enrolled_recall": rep["Enrolled"]["recall"],
                         "Graduate_recall": rep["Graduate"]["recall"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "sensitivity_fee_debtor.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
