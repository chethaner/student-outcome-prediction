"""Leakage-safe preprocessing + model definitions.

Every learned step (imputer statistics, scaler mean/std, one-hot categories) lives
INSIDE the sklearn Pipeline, so it is fitted on training data only - also inside each
cross-validation fold - and re-used unchanged at prediction time.
Class weighting is an estimator parameter, i.e. applied during training only.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_utils import BINARY_FEATURES, NOMINAL_FEATURES, NUMERIC_FEATURES, SEED


def build_preprocessor(exclude: tuple = ()) -> ColumnTransformer:
    """`exclude` drops extra features (used only by the sensitivity analysis)."""
    keep = lambda cols: [c for c in cols if c not in exclude]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # safety net; UCI reports no NaNs
        ("scale", StandardScaler()),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # categories seen < 10 times in training are pooled; unseen codes at predict time are tolerated
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])
    binary = SimpleImputer(strategy="most_frequent")  # already 0/1, no scaling needed
    return ColumnTransformer(
        [("num", numeric, keep(NUMERIC_FEATURES)),
         ("bin", binary, keep(BINARY_FEATURES)),
         ("cat", nominal, keep(NOMINAL_FEATURES))],
        remainder="drop",
    )


def _pipe(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def build_models() -> dict:
    """name -> (pipeline, hyper-parameter grid searched with CV on the TRAINING set only)."""
    return {
        "Dummy (most frequent)": (
            _pipe(DummyClassifier(strategy="most_frequent")), {}),
        "Logistic Regression (unweighted)": (
            _pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
            {"clf__C": [0.01, 0.1, 1.0]}),
        "Logistic Regression (balanced)": (
            _pipe(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
            {"clf__C": [0.01, 0.1, 1.0]}),
        "Random Forest (balanced)": (
            _pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                         n_jobs=1, random_state=SEED)),
            {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 5, 10]}),
    }
