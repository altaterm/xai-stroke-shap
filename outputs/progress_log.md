# Progress log

## Phase 0: Setup and checks (2026-11-07) - done
- CSV loads: 5110 rows, 12 columns. bmi has 201 missing values. stroke: 4861 zero, 249 one.
- Real spelling in the CSV is `Govt_job` (template says `Govt_jov`).
- Template opens; placeholder list saved in outputs/template_placeholders.txt.
- PROBLEM: shap, imbalanced-learn, python-docx and python-pptx were not installed anywhere. The plan says not to install, so I asked the user. Network was on and the user approved installing them in a project venv (.venv, git-ignored). System Python was not changed.
- Versions used are in requirements.txt (pip freeze of the venv). The venv resolved newer versions than system Python (scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, matplotlib 3.11.2, shap 0.54.1).
- All imports OK with .venv/Scripts/python.

## Phase 1: Task 1 EDA (2026-11-07) - done
- Ran `python 00_data_exploration.py` unchanged from the project root with the venv Python, with MPLBACKEND=Agg set in the environment so plt.show() does not block. Exit code 0, no errors, no fixes needed. Output in outputs/eda/00_data_exploration_stdout.txt.
- work/01_eda.py wrote eda_facts.json, missing_values.csv, numeric_summary.csv, categorical_summary.csv and figure1_eda_panel.png.
- No duplicate rows or ids. Observations in outputs/eda/observations.md.
- Glucose unit is not stated in the source, so the figure says so instead of assuming mg/dL.

## Phase 2: Task 2 model and explainer (2026-11-07) - done
- Split 80/20 stratified with random_state=2021: 4088 train (199 positive), 1022 test (50 positive). Train oversampled to 7778 (3889 each class), test untouched.
- bmi median from train = 28.1. 21 encoded features. The gender Other row landed in train.
- RandomForest at threshold 0.5: accuracy 0.940, balanced accuracy 0.513, precision 0.133, recall 0.040, F1 0.062, ROC AUC 0.827, AP 0.145. TN 959, FP 13, FN 48, TP 2.
- PROBLEM reported, not fixed: recall is very low at 0.5. Threshold was not tuned, as the plan says to use 0.5 unless there is a stated reason.
- shap 0.54.1 returns one array (1022, 21, 2), not a list. classes_ = [0, 1], stroke = 1 is index 1.
- Reconstruction check passes: max residual 3.28e-15, tolerance 1e-6.
- Base value is 0.5005 because the trees were trained on the balanced oversampled data. Test prevalence is 4.89%, so the scale is not a prevalence-scale risk.

## Phase 3: Task 3 local explanation (2026-11-07) - done
- Rule: the true-positive test case with the highest predicted probability. Only 2 TP cases exist (test_row 90 at 0.58, test_row 413 at 0.72), so this is a favourable case and the report says so.
- Case test_row 413: actual 1, predicted 1, probability 0.72, base 0.5005, residual 5.6e-16.
- Top 3: heart_disease=1 (+0.163), bmi=36.6 (-0.087), avg_glucose_level=228.69 (+0.057). Remaining features +0.087.
- First force plot labels overlapped, so only the largest are now labelled (contribution_threshold=0.15). All values are in the CSV and HTML.
