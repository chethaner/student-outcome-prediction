import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_synthetic_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Tiny fake dataset with the UCI schema - ONLY for tests/smoke runs, never for results."""
    r = np.random.default_rng(seed)
    d = {
        "Marital status": r.integers(1, 7, n), "Application mode": r.choice([1, 8, 17, 39, 44], n),
        "Application order": r.integers(0, 7, n), "Course": r.choice([33, 171, 8014, 9003, 9070, 9085], n),
        "Daytime/evening attendance": r.integers(0, 2, n), "Previous qualification": r.choice([1, 2, 3, 19, 39], n),
        "Previous qualification (grade)": r.normal(132, 13, n).round(1), "Nacionality": r.choice([1, 1, 1, 1, 41, 62], n),
        "Mother's qualification": r.choice([1, 19, 34, 37, 38], n), "Father's qualification": r.choice([1, 19, 34, 37, 38], n),
        "Mother's occupation": r.choice([0, 4, 5, 7, 9, 90], n), "Father's occupation": r.choice([0, 4, 5, 7, 9, 90], n),
        "Admission grade": r.normal(127, 14, n).round(1), "Displaced": r.integers(0, 2, n),
        "Educational special needs": (r.random(n) < 0.02).astype(int), "Debtor": (r.random(n) < 0.12).astype(int),
        "Tuition fees up to date": (r.random(n) < 0.88).astype(int), "Gender": r.integers(0, 2, n),
        "Scholarship holder": (r.random(n) < 0.25).astype(int), "Age at enrollment": r.integers(17, 60, n),
        "International": (r.random(n) < 0.02).astype(int),
        "Unemployment rate": r.choice([7.6, 10.8, 12.4, 13.9, 15.5], n), "Inflation rate": r.choice([-0.8, 0.3, 1.4, 2.6], n),
        "GDP": r.choice([-4.1, 0.79, 1.74, 3.51], n),
    }
    for sem in ("1st", "2nd"):
        for s in ("(credited)", "(enrolled)", "(evaluations)", "(approved)", "(grade)", "(without evaluations)"):
            d[f"Curricular units {sem} sem {s}"] = r.integers(0, 12, n)
    df = pd.DataFrame(d)
    score = (df["Admission grade"] - 127) / 14 - 1.2 * df["Debtor"] + 0.5 * df["Scholarship holder"] + r.normal(0, 1, n)
    df["Target"] = np.where(score < -0.6, "Dropout", np.where(score < -0.1, "Enrolled", "Graduate"))
    return df


@pytest.fixture(scope="session")
def df():
    return make_synthetic_df()
