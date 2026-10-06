"""Data loading and feature definitions shared by EDA, training, prediction and tests.

Prediction-time rule (assignment): the primary model may only use information
available at enrollment. All 12 "Curricular units 1st/2nd sem (...)" columns are
therefore NEVER used as features. Features are selected by an explicit allow-list
(FEATURES), so a prohibited column cannot slip in by accident.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_pipeline.joblib"

TARGET = "Target"
PROHIBITED_PREFIX = "Curricular units"  # the 12 semester-performance columns

# --- Permitted enrollment-time features (24 of the 36 UCI inputs) -------------
NUMERIC_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
# Integer-coded categories (codes carry NO order) -> one-hot encoded
NOMINAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",  # (sic) spelling used by the UCI dataset
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the semicolon-delimited panel file. Only header whitespace is stripped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download data.csv from the panel's Google "
            "Drive folder and place it in the data/ folder (see README)."
        )
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()  # e.g. 'Daytime/evening attendance\t'
    return df


def prohibited_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(PROHIBITED_PREFIX)]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the permitted enrollment-time predictors (validated)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Expected feature columns missing from data: {missing}")
    X = df[FEATURES].copy()
    assert not any(c.startswith(PROHIBITED_PREFIX) for c in X.columns)
    return X


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    return select_features(df), df[TARGET].copy()


def make_split(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split with a fixed seed. The test set is used ONCE, at the end."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
