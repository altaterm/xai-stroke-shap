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

## Phase 4: Task 4 global explanation (2026-11-07) - done
- Whole test set explained (1022 rows, 50 positive), no sampling. 15 of 21 encoded features displayed. Columns used as encoded, no grouping.
- Top 5 by mean |SHAP|: age 0.165, avg_glucose_level 0.093, bmi 0.082, hypertension 0.021, heart_disease 0.014.
- Task 3 case is atypical: age ranks 15th for that case, and its heart_disease and glucose SHAP values are the test-set maxima.
- Directions for the top-5 table were measured (global_summary.json), not read by eye.

## Phase 5: Task 6 different black-box model (2026-11-07) - done
- StandardScaler + MLPClassifier(max_iter=500, random_state=2021), same split, preprocessing and oversampling. Scaler fitted on training data inside the model pipeline.
- Accuracy 0.876, balanced accuracy 0.555, precision 0.103, recall 0.200, F1 0.136, ROC AUC 0.725, AP 0.124. TN 885, FP 87, FN 40, TP 10.
- PROBLEM reported: ConvergenceWarning, 500 iterations reached. Not tuned.
- Kernel SHAP: background 100 rows from oversampled train (seed 2021), 200 explained rows including the Task 3 case (14 positive), nsamples 2048, l1_reg False, about 37 s, no explainer warnings, max residual 3.3e-15.
- Same case: MLP output 0.959, also predicted 1. Signs match on 11 of 21 features, bmi flips.
- Optional second background (seed 2022) was quick so it was run: top 5 identical, Spearman 0.94, base value moved 0.522 to 0.623, 2 of 21 local signs changed.

## Phase 6: Report (2026-11-07) - done, final rebuild after the slides
- work/07_build_report.py fills the template from outputs/ and refuses to save if any template placeholder survives. Abstract 181 words, conclusion 115 words.
- Student fields stay as highlighted [Student: ...] markers.
- Extra finding from checking a prose claim: positive bmi SHAP values sit at bmi 27.9 to 28.1, next to the imputed 28.1, and 8 of 12 are imputed rows. Reported as possible missingness signal in Sections 5 and 8.2.
- Rendered through Word. LibreOffice and poppler are not installed, so pages were rendered with the Windows PDF API from a scratch script.

## Phase 7: Task 5 slides (2026-11-07) - done
- work/08_build_slides.py builds exactly 3 slides with python-pptx, reading the same evidence files as the report. Plain white, Calibri, no decoration.
- Patient chart is built from the same SHAP values, with words and no numbers.
- Rendered with PowerPoint and checked by eye. Fixed a clipped label on slide 3 and enlarged the force plot on slide 2.

## Phase 8: Final verification (2026-11-07) - done
- Deleted all generated outputs and re-ran work/run_all.py from scratch. All metrics, the case and the top features matched. Only timings changed (the report takes those from the JSON files, so it updates itself). Summary-plot jitter is now seeded.
- work/09_final_checks.py passes (results in outputs/final_checks.json): train and test disjoint, no test row in the oversampled train set, bmi median from train, id and stroke not predictors, 9 tables and 5 figures present, headings kept apart from "How to use this template", exactly 3 slides, slide numbers match the report, no banned words.
- Wording scan: every causal, risk and diagnosis hit is a negation or a statement about what the work does not claim.
- Original course scripts, data and template are unchanged from the baseline commit.
- Report is 15 pages in Word with no blank pages.
- Checklist: Tasks 2 to 6 and the integrity item are ticked. Task 1 and "All placeholders replaced" stay unticked until the student enters the repository details and personal fields.
