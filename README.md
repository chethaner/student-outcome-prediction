# Student Academic Outcome Prediction — Gupio AI/ML Assignment (Option 1)

Predict whether a student ends as **Dropout**, **Enrolled** or **Graduate** using **only information available at enrollment**.

> **Status:** results below were produced by running `python -m src.train` on the supplied dataset
> (Python 3.12.3, scikit-learn 1.8.0, seed 42). **Re-run before submitting and confirm every number matches `outputs/`.**
> Sections still marked `TODO` need your own input.

## 1. Dataset
- UCI ML Repository — *Predict Students' Dropout and Academic Success* (ID 697), 4,424 students, 36 input features + `Target`.
  <https://archive.ics.uci.edu/dataset/697/predict-students-dropout-and-academic-success> · DOI 10.24432/C5MC89 · CC BY 4.0
- File used: the panel-supplied `data.csv` (semicolon-delimited), placed at `data/data.csv`. **Data values are never modified**;
  only header whitespace is stripped when loading. Nothing is derived/added to the features.

## 2. Technologies
Python 3, pandas, NumPy, scikit-learn, matplotlib, joblib, pytest, Jupyter.

## 3. Setup & run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the supplied file at data/data.csv, then (from the project root):
python -m src.train        # EDA + CV model selection + one-time test evaluation + saves models/final_pipeline.joblib
python -m src.predict      # sample predictions with the saved pipeline (same preprocessing)
python -m pytest -q        # unit tests (leakage / split / pipeline checks)
```
Notebook alternative: open `notebooks/01_student_outcome_workflow.ipynb` → *Restart & Run All*.
Outputs (metrics CSV, per-model classification reports, confusion matrices, EDA figures, importances) go to `outputs/`.

```
data/        supplied dataset (data.csv)           src/data_utils.py  loading, feature allow-list, split
notebooks/   EDA + modelling notebook              src/eda.py         overview text + EDA figures
src/         reusable code                         src/pipeline.py    preprocessing + model definitions
models/      saved pipeline (git-ignored)          src/train.py       CV selection, test evaluation, importance
                                           src/sensitivity.py fee/debtor sensitivity check (diagnostic only)
outputs/     figures and result files              src/predict.py     sample-prediction demo
tests/       pytest unit tests
```

## 4. Prediction-time rule & preprocessing decisions
- **Primary model uses enrollment-time information only.** All 12 `Curricular units 1st/2nd sem (...)` columns are excluded
  (features are chosen from an explicit allow-list of 24 columns; a unit test asserts none of the prohibited columns can enter).
- Feature groups: 7 numeric (scaled), 8 binary 0/1 (kept as is), 9 integer-coded nominal (e.g. `Course`, `Application mode`,
  occupations/qualifications → one-hot; codes have no numeric order).
- Median / most-frequent imputers are included as a safety net (UCI reports no missing values).
- One-hot encoder pools categories seen < 10 times in training and tolerates unseen codes at prediction time.
- All learned steps (imputer, scaler, encoder) sit **inside one sklearn `Pipeline`**, so they are fitted on training data only
  — including inside each CV fold — and the identical fitted pipeline is used for prediction.
- `Debtor` and `Tuition fees up to date` are kept in the primary model because the assignment only prohibits the 12 semester variables, but they
  may be recorded *after* enrollment (a student who is about to drop out stops paying) — a possible soft leak. A sensitivity check is reported in section 7.
- Data-quality observations (`outputs/data_overview.txt`): 4,424 rows × 37 columns; 0 missing values; 0 duplicate rows; all columns numeric
  (integer codes for categorical variables); both grade columns range 95–190 (no zeros/invalid values); `Age at enrollment` 17–70
  (right-skewed, median 20); many integer-coded categories are very rare (e.g. 32 of 46 `Father's occupation` codes have < 10 students),
  which is why rare categories are pooled by the encoder. `Nacionality` is the dataset's own spelling.

## 5. Train / test strategy
- Stratified 80/20 split, `random_state=42`. The 20 % test set is untouched until models are chosen.
- Model selection: 5-fold stratified CV on the **training set**, metric = **macro-F1**; small hyper-parameter grids per model
  (`GridSearchCV`, training data only). Selection is made *before* any test-set scoring; each model is then scored once on the test set.
- Class imbalance: `class_weight` (balanced / balanced_subsample) is applied only inside training. An unweighted logistic
  regression is included to show the effect of weighting.

## 6. Models compared
Dummy (most frequent) floor · Logistic Regression (unweighted) · Logistic Regression (balanced) · Random Forest (balanced).

## 7. Results (actual output of `python -m src.train`)
Train = 3,539 students, held-out test = 885. Class shares: Graduate 49.9 %, Dropout 32.1 %, Enrolled 18.0 %.

| Model | CV macro-F1 (train) | Test accuracy | Test macro-F1 | Dropout F1 | Enrolled F1 | Graduate F1 |
|---|---|---|---|---|---|---|
| Dummy (most frequent) | 0.222 | 0.499 | 0.222 | 0.000 | 0.000 | 0.666 |
| Logistic Regression (unweighted) | 0.554 | 0.628 | 0.520 | 0.624 | 0.203 | 0.731 |
| Logistic Regression (balanced) | 0.574 | 0.583 | 0.560 | 0.633 | 0.393 | 0.654 |
| **Random Forest (balanced)** — selected | **0.585** | 0.593 | 0.541 | 0.619 | 0.325 | 0.680 |

(Full per-class precision/recall in `outputs/model_comparison.csv` and `outputs/classification_report_*.txt`.)

Per-class precision / recall / F1 and confusion matrices: `outputs/` and `outputs/figures/cm_*.png`.

**Final model: Random Forest (balanced)** (`min_samples_leaf=5`, no depth limit, 300 trees). It was chosen by the rule fixed *before* looking at
the test set: highest 5-fold CV macro-F1 on the training data (0.585 vs 0.574 for balanced logistic regression). Accuracy was not used —
the unweighted logistic regression has the highest test accuracy (0.628) but the worst Enrolled recall (0.15).
Honest note: on the single test split, balanced logistic regression scores slightly higher (macro-F1 0.560 vs 0.541) and finds far more
Enrolled students (recall 0.54 vs 0.34). The CV gap between the two models (0.011) is small, so the two are close; I did **not** switch the
model after seeing test scores, because that would turn the test set into a selection set. If the use case valued catching Enrolled/at-risk students
over overall precision, balanced logistic regression would be a defensible alternative.

**Class imbalance:** Graduate is ~50 % of the data and Enrolled only ~18 %. Without class weights the models favour the majority classes
(unweighted LR: Graduate recall 0.83, Enrolled recall 0.15 — Enrolled F1 0.20). Class weighting raises Enrolled recall (RF 0.34, LR 0.54) at the cost of
Graduate recall and accuracy. For the selected RF (test set): Dropout recall 0.64 / precision 0.60 and Graduate recall 0.66 / precision 0.71 are
moderate; **Enrolled is handled poorly** (precision 0.31, recall 0.34) — 40 % of Enrolled students are predicted Graduate and 26 % Dropout
(`outputs/figures/cm_random_forest_balanced.png`). Enrolled is a transitional state without a distinct enrollment-time profile, and
enrollment-only information is limited, so overall performance is modest (accuracy ≈ 0.59, macro-F1 ≈ 0.54; a dummy baseline gets 0.22).

**Sensitivity check (`python -m src.sensitivity`, test macro-F1):** all 24 features RF 0.541 / LR 0.560 → without `Tuition fees up to date` RF 0.530 / LR 0.533
→ without it and `Debtor` RF 0.512 / LR 0.519.

**Sample predictions:** `outputs/sample_predictions.csv` (2 held-out students per class: actual, predicted, class probabilities).

## 8. Interpretation
Permutation importance of the selected RF on the test set (drop in macro-F1 when a feature is shuffled; `outputs/permutation_importance.csv`):
`Tuition fees up to date` (0.043) ≫ `Course` (0.018) > `Scholarship holder` (0.016) > `Mother's occupation` (0.009) ≈ `Admission grade` (0.009); the rest are
≤ 0.005 and several are within one standard deviation of zero. This agrees with the EDA: 86.6 % of students whose fees are *not* up to date
dropped out (vs 24.7 %), scholarship holders graduate far more often (76 % vs 41 %), debtors drop out more (62 % vs 28 %), and dropout varies
strongly by course (e.g. 15 % for course 9500 vs 54 % for course 9119).
Importance reflects **predictive association, not causation** (e.g. a high-importance feature does not mean changing it would change a student's outcome).

## 9. Assumptions & limitations
- Prediction is made at enrollment; semester-performance variables (which are far more predictive) are deliberately not used.
- Single institution / time period → may not generalise to other institutions or cohorts.
- `Enrolled` is the weakest class (F1 ≈ 0.33 for the selected model, see section 7).
- **Possible soft leakage:** `Tuition fees up to date` is by far the most important feature. Dropping it (and `Debtor`) lowers test macro-F1 from 0.541 to 0.512 (RF) and 0.560 to 0.519 (LR) — see `outputs/sensitivity_fee_debtor.csv` (diagnostic only, not used for selection). The model is still above the baseline without them, but the primary-model figures are likely optimistic for a true day-of-enrollment prediction.
- Predictions are statistical associations and must not be used to make high-stakes decisions about individual students;
  fairness concerns apply (e.g. `Gender`, `Nacionality`, `International` are present in the data).
- Permutation importance is computed on one test split and is affected by correlated features.
- No cross-validated estimate of the *final* test score variability is reported (single hold-out split).

## 10. Completed / incomplete features
- [x] Data inspection, EDA, leakage-safe pipeline, stratified split, ≥2 models, per-class metrics, macro-F1, confusion matrices
- [x] Sample predictions, feature interpretation, unit tests (bonus), cross-validated grid search (bonus)
- [ ] TODO: list anything you did not finish (e.g. SHAP, inference API, Docker) — do not claim what is not done.

## 11. What I verified before submitting
TODO — tick only what you really did:
- [ ] Ran in a fresh virtual environment from step 1 of *Setup* to the end
- [ ] `python -m src.train`, `python -m src.predict`, `python -m pytest -q` all succeed
- [ ] Numbers in section 7 match `outputs/model_comparison.csv`
- [ ] No credentials / `.env` / API keys in the repository

## 12. AI-assistance disclosure
TODO — state honestly how you used AI (e.g. "Used Claude chat as a coding/reference assistant; I reviewed and understand all code.
No autonomous coding agents were used.").

## Data licence
Dataset: CC BY 4.0, Realinho et al. (2021), UCI Machine Learning Repository.
