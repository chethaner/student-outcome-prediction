# Student Academic Outcome Prediction

Predict whether a student ends as **Dropout**, **Enrolled**, or **Graduate** using only information available at the time of enrollment.

---

## 1. Dataset Source

- **Dataset:** UCI ML Repository — *Predict Students' Dropout and Academic Success* (ID 697)
- **Size:** 4,424 students · 36 input features + `Target`
- **Link:** <https://archive.ics.uci.edu/dataset/697/predict-students-dropout-and-academic-success>
- **DOI:** 10.24432/C5MC89 · **License:** CC BY 4.0 · Realinho et al. (2021)
- **File used:** `data/data.csv` (semicolon-delimited, panel-supplied). Data values are never modified — only header whitespace is stripped on load.

---

## 2. Technologies

| Layer | Libraries |
|---|---|
| Language | Python 3.12 |
| Data | pandas, NumPy |
| ML | scikit-learn 1.8.0 |
| Serialisation | joblib |
| Visualisation | matplotlib |
| Testing | pytest |
| Notebook | Jupyter |

---

## 3. Setup & Run Instructions

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
#    Copy supplied file → data/data.csv

# 4. Run training (EDA + CV model selection + test evaluation + saves pipeline)
python -m src.train

# 5. Run sample predictions using the saved pipeline
python -m src.predict

# 6. Run unit tests
python -m pytest -q
```

**Notebook alternative:** Open `notebooks/01_student_outcome_workflow.ipynb` → *Restart & Run All*.

All outputs (metrics CSV, classification reports, confusion matrices, EDA figures, feature importances) are saved to `outputs/`.

### Project Structure

```
data/           supplied dataset (data.csv)
notebooks/      EDA + modelling notebook
src/            reusable source code
  data_utils.py   loading, feature allow-list, train/test split
  eda.py          EDA overview text + figures
  pipeline.py     preprocessing + model definitions
  train.py        CV selection, test evaluation, feature importance
  sensitivity.py  fee/debtor sensitivity check (diagnostic only)
  predict.py      sample-prediction demo
models/         saved pipeline (git-ignored)
outputs/        figures and result files
tests/          pytest unit tests
```

---

## 4. Preprocessing Decisions

- **Enrollment-time features only.** All 12 `Curricular units 1st/2nd sem (...)` columns are excluded. Features are chosen from an explicit allow-list of 24 columns; a unit test asserts none of the prohibited columns can enter the pipeline.
- **Feature groups:**
  - 7 numeric → StandardScaler
  - 8 binary (0/1) → kept as-is
  - 9 integer-coded nominal (e.g. `Course`, `Application mode`, occupations/qualifications) → OneHotEncoder
- **Imputation:** Median (numeric) and most-frequent (categorical) imputers included as a safety net (UCI reports 0 missing values).
- **Rare category pooling:** Categories seen fewer than 10 times in training are pooled; unseen codes at prediction time are handled gracefully.
- **No leakage:** All learned steps (imputer, scaler, encoder) are inside a single `sklearn.Pipeline` — fitted on training data only, including inside each CV fold.
- **Soft-leakage note:** `Tuition fees up to date` and `Debtor` are retained in the primary model because only the 12 semester variables are prohibited. However, these fields may be recorded *after* enrollment (a student who is about to drop out stops paying). A sensitivity check is reported in the Results section.

---

## 5. Train / Test Strategy

| Setting | Value |
|---|---|
| Split | Stratified 80 / 20 |
| Random state | 42 |
| CV folds | 5-fold stratified (on training set only) |
| Selection metric | Macro-F1 |
| Hyper-parameter search | `GridSearchCV` (training data only) |
| Class imbalance | `class_weight="balanced"` / `"balanced_subsample"` inside training |

The 20 % test set is **held out entirely** until after model selection. Each model is scored on the test set exactly once.

---

## 6. Prediction-Time Assumptions

- Prediction is made **at the point of enrollment** — no semester-level academic performance data is used.
- All 24 features in the allow-list must be available at enrollment time.
- The saved pipeline (`models/final_pipeline.joblib`) applies identical preprocessing to new data; no separate scaling or encoding step is needed.
- `Tuition fees up to date` and `Debtor` are assumed to reflect enrollment-time status. If they are updated later, the model may be optimistic (see sensitivity check).

---

## 7. Models Compared

| Model | Description |
|---|---|
| Dummy (most frequent) | Always predicts Graduate — baseline floor |
| Logistic Regression (unweighted) | Default class weights |
| Logistic Regression (balanced) | `class_weight="balanced"` |
| **Random Forest (balanced)** | `balanced_subsample`, 300 trees, `min_samples_leaf=5` — **selected** |

---

## 8. Results

**Split:** Train = 3,539 students · Test = 885 students  
**Class shares:** Graduate 49.9 % · Dropout 32.1 % · Enrolled 18.0 %

| Model | CV macro-F1 (train) | Test accuracy | Test macro-F1 | Dropout F1 | Enrolled F1 | Graduate F1 |
|---|---|---|---|---|---|---|
| Dummy (most frequent) | 0.222 | 0.499 | 0.222 | 0.000 | 0.000 | 0.666 |
| Logistic Regression (unweighted) | 0.554 | 0.628 | 0.520 | 0.624 | 0.203 | 0.731 |
| Logistic Regression (balanced) | 0.574 | 0.583 | 0.560 | 0.633 | 0.393 | 0.654 |
| **Random Forest (balanced)** ✅ | **0.585** | 0.593 | 0.541 | 0.619 | 0.325 | 0.680 |

Full per-class precision/recall in `outputs/model_comparison.csv` and `outputs/classification_report_*.txt`.  
Confusion matrices: `outputs/figures/cm_*.png`.

**Selected model:** Random Forest (balanced) — chosen by the rule fixed *before* seeing the test set: highest 5-fold CV macro-F1 on training data (0.585 vs 0.574 for balanced LR). Accuracy was not the selection criterion; the unweighted LR has the highest test accuracy (0.628) but very poor Enrolled recall (0.15).

**Sensitivity check (drop in test macro-F1 when features removed):**

| Features | RF | LR |
|---|---|---|
| All 24 (primary model) | 0.541 | 0.560 |
| Without `Tuition fees up to date` | 0.530 | 0.533 |
| Without fees + `Debtor` | 0.512 | 0.519 |

**Feature importance (permutation, test set):**  
`Tuition fees up to date` (0.043) ≫ `Course` (0.018) > `Scholarship holder` (0.016) > `Mother's occupation` (0.009) ≈ `Admission grade` (0.009). The rest are ≤ 0.005. Full table: `outputs/permutation_importance.csv`.

**Sample predictions:** `outputs/sample_predictions.csv` (2 held-out students per class: actual, predicted, class probabilities).

---

## 9. Assumptions & Limitations

- **Enrollment-time only:** Semester-performance variables (which are far more predictive) are deliberately excluded.
- **Single institution/cohort:** Results may not generalise to other institutions or time periods.
- **Enrolled class is weak:** F1 ≈ 0.33 for the selected model. Enrolled is a transitional state without a distinct enrollment-time profile — 40 % of Enrolled students are predicted Graduate and 26 % Dropout.
- **Possible soft leakage:** `Tuition fees up to date` is the most important feature and may be recorded after enrollment. Primary-model figures are likely optimistic for a true day-of-enrollment prediction (sensitivity check shows F1 drops from 0.541 → 0.512 when removed).
- **Fairness:** `Gender`, `Nacionality`, and `International` are present in the data. Predictions are statistical associations and must not be used to make high-stakes decisions about individual students.
- **Importance caveat:** Permutation importance is computed on one test split and is affected by correlated features. It reflects predictive association, not causation.
- **Variability:** No cross-validated estimate of final test score variability is reported (single hold-out split).

---

## 10. Completed / Incomplete Features

### ✅ Minimum Deliverables

- [x] **Working notebook and source code** — complete workflow in `notebooks/01_student_outcome_workflow.ipynb` and `src/`
- [x] **Reproducible preprocessing and modelling code** — runs end-to-end with `python -m src.train` without manual steps
- [x] **Evaluation output** — model comparison table, per-class metrics, macro-F1, confusion matrices in `outputs/`
- [x] **Sample prediction demonstration** — `python -m src.predict` → `outputs/sample_predictions.csv`
- [x] **README.md** — dataset source, setup, run instructions, preprocessing decisions, train/test strategy, prediction-time assumptions, models, results, assumptions, limitations, completed/incomplete features
- [x] **requirements.txt** — all dependencies pinned
- [x] **.gitignore** — models, cache, virtual env, secrets excluded

### ✅ Quality Checks

- [x] Train and test records are strictly separated (stratified 80/20 split, no overlap)
- [x] Encoders, scalers, imputers are fitted on training data only — inside `sklearn.Pipeline`, including within each CV fold
- [x] Prohibited semester-performance variables (`Curricular units 1st/2nd sem (...)`) are excluded via explicit allow-list; unit test asserts this
- [x] Model selected by CV macro-F1 on training data before any test-set scoring; test set scored exactly once
- [x] All model scores, predictions and charts are produced by running the code on the supplied dataset — none fabricated
- [x] Dataset values unchanged — only header whitespace stripped on load; documented in section 4

### ✅ Bonus Opportunities

- [x] **Cross-validation** — 5-fold stratified `GridSearchCV` for hyperparameter tuning
- [x] **Systematic hyperparameter tuning** — per-model grids (LR: C, penalty; RF: n_estimators, min_samples_leaf)
- [x] **Additional classification metrics** — per-class precision, recall, F1; confusion matrices; permutation feature importance
- [x] **Unit tests** — leakage check, split integrity, pipeline reproducibility (`tests/test_pipeline.py`)
- [ ] Model explainability (SHAP)
- [ ] Inference API
- [ ] Interactive demo

---

## Data Licence

Dataset: CC BY 4.0 — Realinho et al. (2021), UCI Machine Learning Repository.
