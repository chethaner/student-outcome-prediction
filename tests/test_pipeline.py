import numpy as np
import pandas as pd

from src.data_utils import (FEATURES, NUMERIC_FEATURES, PROHIBITED_PREFIX, make_split,
                            split_features_target)
from src.pipeline import build_models, build_preprocessor


def test_prohibited_semester_columns_never_used(df):
    X, _ = split_features_target(df)
    assert len(FEATURES) == 24
    assert list(X.columns) == FEATURES
    assert not any(c.startswith(PROHIBITED_PREFIX) for c in X.columns)


def test_split_is_disjoint_and_stratified(df):
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = make_split(X, y)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(X)
    full, test = y.value_counts(normalize=True), y_te.value_counts(normalize=True)
    assert np.allclose(full.sort_index().values, test.sort_index().values, atol=0.02)


def test_preprocessing_is_fitted_on_training_data_only(df):
    X, y = split_features_target(df)
    X_tr, X_te, _, _ = make_split(X, y)
    prep = build_preprocessor().fit(X_tr)
    mean = prep.named_transformers_["num"].named_steps["scale"].mean_
    assert np.allclose(mean, X_tr[NUMERIC_FEATURES].mean().values)
    assert not np.allclose(mean, X[NUMERIC_FEATURES].mean().values)  # would differ if test leaked in


def test_pipeline_predicts_and_tolerates_unseen_category(df):
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, _ = make_split(X, y)
    pipe, _ = build_models()["Logistic Regression (balanced)"]
    pipe.fit(X_tr, y_tr)
    row = X_te.iloc[[0]].copy()
    row["Course"] = 99999  # unseen code
    proba = pipe.predict_proba(row)
    assert proba.shape == (1, 3) and np.isclose(proba.sum(), 1.0)
