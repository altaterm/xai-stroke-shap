# Final update: threshold selection, polish, and submission-ready report

The assignment (Stroke prediction with SHAP) is already complete: scripts in `work/`, results in `outputs/`, `Stroke_SHAP_Report_Completed.docx`, and `outputs/slides/Stroke_SHAP_Slides.pptx`. Everything was built by `work/run_all.py` and all numbers come from the files in `outputs/`.

This task makes **one substantive improvement** (a justified decision threshold) and **two small polish fixes**, then fills in the student's personal details so the report can be submitted. Do nothing else.

## Why the threshold change

Both models currently use a hardcoded threshold of 0.5 and no selection rule. The report template has a field "Decision threshold and selection rule: select using training or validation data". The Random Forest catches only 2 of 50 strokes at 0.5 (recall 0.040). The fix is to choose the threshold from **out-of-fold predictions on the training set only**, then apply it once to the test set.

## Scope rules (strict)

1. **Do only what is listed below.** No hyperparameter tuning, no class weights, no SMOTE, no calibration, no new models, no new explainers, no extra plots, no extra slides, no new report sections.
2. **The final trained models and all SHAP results must not change.** The Random Forest, the MLP, the SHAP values, the summary plots, the force plots and the selected case (test_row 413) stay as they are. Only the threshold, and the numbers and text that depend on it, change.
3. **Never use the test set to choose or adjust the threshold or the rule.** The rule below is fixed now. Do not change it after seeing test results.
4. **Do not edit the original course files** (`00`-`05` scripts, `utils.py`, the blank template).
5. **Do not install anything.** Everything needed (scikit-learn, imbalanced-learn, shap) is already installed.
6. **Every number and claim in the report and slides must come from code that was run.** Rewrite any sentence that stops being true with the new results (for example "recall is tiny", "missed 48 of 50").
7. Slide style rules from the original plan still apply: exactly three slides, plain white, no decoration, short student-style bullets, none of these words: leverage, robust, delve, seamless, key takeaway, at a glance, and no em dashes.
8. Keep claims modest: associations not causes, no validated future-risk wording, no diagnosis or treatment advice.

## Phase 0: Safety net

1. Make sure the working tree is committed, then create a local git tag `before-threshold-update` so the earlier state can be restored.
2. Copy the current `outputs/task2/metrics.json`, `outputs/task6/metrics.json` and `outputs/task3/selected_case.json` somewhere in the scratchpad. You will compare against them at the end.
3. Record SHA-256 of `outputs/task2/shap_values_test.npz` and `outputs/task6/kernel_shap_values.npz` so you can prove the SHAP results did not change.

## Phase 1: Make preprocessing foldable (no change in behaviour)

The template says imputation, encoding and resampling must be fitted on training data or **training folds only**. Inside cross-validation, the BMI median and the one-hot encoder must therefore be fitted on each fold's training rows.

1. In `work/common.py`, split the fitting logic out of `prepare_data()` into a helper (for example `fit_preprocessing(X_raw_train)` that returns a `transform(frame)` function plus the median and encoder). `prepare_data()` must use it and return **exactly** the same objects and values as before.
2. Prove nothing changed: compare the new `prepare_data()` output with the old one (feature names and order, `X_train`, `X_test`, `X_train_res`, `y_train_res`, `bmi_median`) and assert they are identical. Stop and report if they are not.

## Phase 2: Choose the threshold on training data only

Create `work/02a_select_threshold.py`. It must:

1. Take the **un-oversampled training split** (`X_train_raw`, `y_train` from `prepare_data()`).
2. Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)`.
3. For each fold, for **each of the two models**:
   - fit the preprocessing (BMI median, one-hot encoder) on the fold's training rows only;
   - oversample **only the fold's training rows** with `RandomOverSampler(sampling_strategy="minority", random_state=2021)`;
   - fit the model with exactly the same settings as the final model: `RandomForestClassifier(random_state=2021)` and `StandardScaler + MLPClassifier(max_iter=500, random_state=2021)`;
   - predict stroke = 1 probabilities on the fold's **held-out rows, which are not oversampled**.
4. Combine the held-out predictions into one out-of-fold score per training row, per model.
5. **Selection rule (fixed): choose the threshold that maximises the F2 score for stroke = 1 on the out-of-fold predictions.** F2 weights recall twice as much as precision, which matches the report's own argument that missing a stroke is the more serious error. Search the thresholds returned by `precision_recall_curve`, and break ties by taking the higher threshold.
6. Save `outputs/threshold/thresholds.json` with, per model: the chosen threshold, the rule, number of folds, seed, and the out-of-fold precision, recall, F2, plus the share of training rows flagged at that threshold. Also save a small out-of-fold table or curve if useful (one figure at most, and only if the report will use it; otherwise skip).
7. **Guard:** if the chosen threshold for either model flags more than 50% of the out-of-fold rows (a near "flag everyone" outcome), **stop and ask the user** how to proceed. Do not silently change the rule.
8. Record fit warnings (for example the MLP ConvergenceWarning) in the JSON and mention them in the report if they occur.

## Phase 3: Apply the thresholds

1. `work/02_train_shap.py`: replace the hardcoded `THRESHOLD = 0.5` with the Random Forest threshold read from `outputs/threshold/thresholds.json`. Keep the model, the explainer and the additivity check unchanged. Keep the 0.5 metrics too, saved as a second block (for example `metrics_at_0_5`) so the report can state the before and after in one sentence.
2. `work/05_task6.py`: do the same for the MLP, with its own threshold.
3. Do **not** use the test set to pick anything. Test metrics are computed once at the chosen thresholds.
4. `work/03_task3_local.py`: the selection rule stays "true positive with the highest predicted probability". Check that the selected row is still 413. If it changes, or if row 413 is no longer predicted positive, **stop and ask the user**.
5. Confirm that `shap_values_test.npz` and `kernel_shap_values.npz` hashes are unchanged from Phase 0. If they changed, something went wrong; find out why before continuing.
6. Add the new script to `work/run_all.py`, before `02_train_shap.py`.

## Phase 4: Update the report text and slides

Only the threshold-dependent numbers and wording change.

**Report (`work/07_build_report.py`)**
- Section 3.1 "Decision threshold and selection rule": state the rule (maximise F2 on 5-fold out-of-fold training predictions, preprocessing and oversampling fitted inside each training fold, seed 2021), the chosen threshold for each model, and the out-of-fold recall and precision at that threshold. Replace the old wording "0.5, the default... no validation data were set aside".
- Update every other place that mentions 0.5 or "no tuned threshold": abstract, Section 3 metrics table cells, Section 4 case paragraph, Section 7 metrics text, Section 8 comparison table and text, limitations, conclusion and the Task 5 slide-plan paragraphs.
- Add one short sentence giving the 0.5 results for comparison (recall and precision at 0.5 versus at the chosen threshold).
- Rewrite the conclusion and abstract using the **new** results. Keep the abstract at 150-200 words and the conclusion at 100-150 words, and keep the existing assertions on word counts.
- State plainly what is still true: calibration was not assessed, the scores come from oversampled training data, one test set with 50 positives is noisy, and there is no external validation. Do not claim the model is now good. If recall improves, report the cost in precision and false alarms.
- Check any sentence that assumed 2 of 50 true positives (for example "only 2 true positives exist, so row 413 is unrepresentative") and make it match the new counts.
- Update the "Evidence files" and "Environment" paragraphs to include the new script and `outputs/threshold/`.
- Section 8.2: the mention of "no separate validation set" should now say the threshold came from out-of-fold training predictions.

**Slides (`work/08_build_slides.py`)**
- Replace every hardcoded "0.5" with the chosen Random Forest threshold read from the saved files, including slide 1's metrics bullet and slide 2's "missed X of Y strokes at threshold ..." bullet.
- Slide 2's "BMI of 36.6" is typed into the code. Read it from `selected_case.json` (`raw_values["bmi"]`) instead.
- Re-check that the wording on all three slides is still true with the new numbers (for example "it misses many people" must still hold; change it if it no longer does). The case, predicted score and base value should be unchanged.
- Still exactly three slides, same style rules.

## Phase 5: Layout polish

- In `07_build_report.py`, enlarge Figure 2 (the force plot) so its text is readable, for example width about 6.3 inches, as long as the section does not push a near-empty page. Check the rendered page count and layout.

## Phase 6: Personal details (ask the user)

Use `AskUserQuestion` (or plain questions if unavailable) and **wait for the answers**. Do not invent any of these. Ask for:

1. Full name
2. Student ID
3. Course and programme
4. Lecturer name
5. Submission date
6. How the course code was obtained (cloned or downloaded as ZIP) and its commit or release if known
7. Repository to cite: the private repo `altaterm/xai-stroke-shap` (previous commit 28bf3c7) or another. Mention that it is private, so the lecturer would need access. Ask whether to push the new commit, and push **only if the user says yes**. After the code commit is made, record its commit hash in the report. Do not push the completed report or any file containing personal details unless the user explicitly agrees.
8. Work contributions: individual or group, and if group, names and tasks
9. Tool-use disclosure: show the current suggested wording ("Claude Code was used to run the analysis, write the code and draft this report, and all reported results come from code that was actually run") and ask the user to confirm or edit it according to the course policy
10. Citation style, if the course requires one (otherwise keep the numbered style)

Then:
- Store the answers in a small local file (for example `work/student_info.json`) that `07_build_report.py` reads, replacing every `[Student: ...]` marker. Add that file to `.gitignore` so it is not committed.
- Once a marker is replaced by a real value it should no longer be highlighted yellow.
- If the user leaves anything unanswered, keep its yellow `[Student: ...]` marker and tell the user which ones remain.
- Tick the checklist items "Task 1 ..." and "All placeholders replaced ..." only if they are truly done (script executed and no `[Student:` marker left).

## Phase 7: Verify

1. Run `python work/run_all.py` from the project root from a clean state (delete generated outputs first) and confirm it finishes with exit code 0.
2. Update `work/09_final_checks.py` for the new rule (for example the thresholds file exists, the same case row, SHAP hashes unchanged, no hardcoded 0.5 left in slides or report except where the 0.5 comparison sentence is intended) and make sure it passes.
3. Check there is no leakage in the threshold step: the held-out fold is never oversampled, preprocessing is fitted inside each fold, and the test set is untouched until the final evaluation.
4. Open the final `.docx` and `.pptx`, render them, and look at them: no leftover placeholders other than any unanswered `[Student: ...]` markers, figures readable, three slides, text fits, numbers on slides equal the report.
5. Compare the old and new metrics and write a short before/after table for the summary.

## Final message to the user

Keep it short. Include:
- the chosen thresholds and the rule;
- a small table of recall, precision, F1, balanced accuracy, ROC AUC and average precision at 0.5 and at the chosen threshold, for both models;
- confirmation that the SHAP results, the selected case and the plots are unchanged;
- anything that failed, was approximate, or needs the user's decision;
- which `[Student: ...]` fields, if any, are still unfilled;
- the final file locations: `Stroke_SHAP_Report_Completed.docx`, `outputs/slides/Stroke_SHAP_Slides.pptx`, `work/`, `outputs/`, `requirements.txt`.