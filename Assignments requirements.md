# Assignment 1: Stroke Prediction with SHAP: Execution Plan for Claude Code

You are completing a university assignment end to end. Work through the phases below **in order**. Finish and verify each phase before starting the next. Keep a short running log in `outputs/progress_log.md` (what was done, what was produced, any problems).

## Ground rules (read first)

1. **Do only what is asked.** No extra models, extra explainers (no LIME, counterfactuals, LRP), extra datasets, dashboards, web apps, or bonus analysis. If something is not in this file or the report template, do not do it.
2. **Never invent results.** Every number, metric, plot and statement in the report and slides must come from code you actually ran. If a check fails, report it honestly.
3. **Do not edit the original course scripts** (`00`-`05`, `utils.py`). Copy them or write new scripts instead. Put new work in `work/`.
4. **Do not use `05_lrp.py`, `04_counterfactuals.py`, `02_lime.py`, `01_interpretable_models.py`.** They are not part of this assignment (the brain MRI images are not supplied anyway).
5. **Network may be off.** Do not try to install packages. Check what is already installed (`pip list`), use that, and tell the user clearly which package is missing if one is.
6. **Seeds:** use `random_state=2021` everywhere (it matches `utils.py`).
7. **No unsupported claims.** Describe associations, never causes. Do not call the model output a validated future stroke risk. The model explains a retrospective dataset only. It is not a diagnosis.
8. **Slides must look human-made, not AI-generated** (see Phase 7 for exact rules).

## Files provided

| File | Use |
|---|---|
| `data/healthcare-dataset-stroke-data.csv` | The dataset |
| `utils.py` | `DataLoader` (load, preprocess, split, oversample). Reuse it, but see the known issues below |
| `00_data_exploration.py` | Task 1 (run it as required) |
| `03_shap.py` | Starting point for Tasks 2-4 (RandomForest + `TreeExplainer`) |
| `Stroke_SHAP_Report_Template.docx` | The report to fill in. This is the main deliverable |

## Known issues in the supplied code (handle, do not ignore)

- `utils.py` fills missing `bmi` with `0`. This is implausible. In your own pipeline, impute `bmi` with the **training-set median** (fit on the training split only). Document it in the report.
- `utils.py` preprocesses (imputes) before splitting. The template requires imputation/encoding/resampling to be **fitted on training data only**. Split first, then impute using training statistics. Oversample **only the training set**; the test set stays untouched.
- `03_shap.py` explains only one row and assumes an older SHAP output format (a list of arrays, one per class). Check the actual shape of `shap_values` on the installed SHAP version and verify the class index against `model.classes_`. Do not trust the script's indexing.
- `id` must be excluded from predictors. `stroke` is the target and must not be in the predictors.
- Template note: the brief writes `Govt_jov`. Check the actual spelling in the CSV (it is probably `Govt_job`) and use the real one.
- `smoking_status == "Unknown"` means unavailable information. Do not treat it as "never smoked".

## Output folder layout

```
work/            your scripts (01_eda.py, 02_train_shap.py, 03_task6.py, ...)
outputs/
  eda/           EDA figures and tables
  task2/         metrics, confusion matrix, saved model
  task3/         force plot (png + html)
  task4/         summary plot (png), mean |SHAP| table
  task6/         metrics, local + global plots for the second model
  slides/        the 3-slide file
  progress_log.md
Stroke_SHAP_Report_Completed.docx
requirements.txt (actual versions used)
```

---

## Phase 0: Setup and checks

1. Confirm the CSV exists and loads. Print shape, dtypes, column names.
2. Record Python version and versions of: `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `imbalanced-learn`, and whichever model library you use for Task 6. Save to `requirements.txt`.
3. Check that the report template opens and list its placeholders.
4. Verify these are importable: `shap`, `sklearn`, `imblearn`. If anything is missing, stop and tell the user rather than working around it silently.

**Done when:** versions recorded, data loads, no missing imports.

---

## Phase 1: Task 1, Exploratory Data Analysis

1. Run `00_data_exploration.py` exactly as supplied (from the folder where `utils.py` and `data/` are reachable). Record: command, date, output, and whether it succeeded. If it errors, note the error and the minimal fix, and keep the original file unchanged (fix in a copy).
2. Compute and save the facts needed for the template's data-quality table:
   - rows and columns
   - duplicate rows and duplicate `id`s
   - missing values per column (count and %), including `bmi` and `smoking_status == "Unknown"`
   - stroke class counts and % positive
   - age, avg_glucose_level, bmi: mean ± SD, median, IQR, min, max (flag implausible values, e.g. age under 1, BMI extremes, the single `gender = Other`)
   - category counts and % for every categorical column
3. Make the EDA figures for **Figure 1 (EDA panel)**: class counts, numeric distributions, missingness, and a few comparisons by stroke status (e.g. age, glucose, hypertension, heart disease by stroke). Save as PNG in `outputs/eda/`. Keep them clear and readable.
4. Write down three evidence-based observations (with the figure/table they come from), e.g. class imbalance, missing BMI, age pattern. Describe associations only.

**Done when:** EDA table values, Figure 1 and three observations are saved.

---

## Phase 2: Task 2, Model and SHAP explainer

1. Build the pipeline in `work/02_train_shap.py`:
   - Predictors: all columns except `id` and `stroke`.
   - Split 80/20 with `random_state=2021`, **stratified on stroke**. Record train and test sizes and positive counts.
   - Impute `bmi` using the training median. One-hot encode categoricals (fit on training; keep feature names). No leakage.
   - Handle imbalance by oversampling the **training set only** (as in the course code). State this in the report.
2. Model: `RandomForestClassifier(random_state=2021)` as in `03_shap.py`. This is the Task 2 model. Keep default hyperparameters unless something is broken (do not tune for extras).
3. Evaluate on the untouched test set. Record exactly what the template asks for:
   - accuracy and balanced accuracy
   - precision, recall, F1 for stroke = 1 (and the positive-class support)
   - ROC AUC and average precision (use predicted probabilities)
   - confusion matrix (TN, FP, FN, TP)
   - threshold used (0.5 unless you have a stated reason, chosen on train/validation only)
4. Create the SHAP explainer: `shap.TreeExplainer(model)`. Record:
   - SHAP version and sklearn version
   - class index for stroke = 1 (**verify with `model.classes_`**)
   - the output explained (probability of stroke = 1 for a RandomForest TreeExplainer) and its scale
   - background: state what is used (TreeExplainer with default tree-path-dependent needs none; if you pass background data, use a training sample and record size and seed)
   - explanation sample: held-out test rows
5. **Verification (template table):** check that `base value + sum(SHAP values)` reconstructs the model's predicted probability for the explained class. Report the max absolute residual and the tolerance. Check feature names, order and dimensions match.
6. Save the trained model, metrics (CSV/JSON) and the confusion matrix figure.

**Done when:** metrics table, verification table and saved explainer inputs exist and the reconstruction check passes (or the failure is documented honestly).

---

## Phase 3: Task 3, Local explanation (force plot)

1. Select **one** held-out case. State the selection rule in the report (e.g. "the true-positive test case with the highest predicted probability" or similar). Do not cherry-pick favourable examples without saying so. Record: anonymous row reference (test-set row index, not `id`), actual class, predicted class, threshold, predicted probability, base value, case type (TP/FP/TN/FN).
2. Produce the SHAP **force plot** for that case for the stroke = 1 output. Export a **static** PNG (use `matplotlib=True` or a screenshot-equivalent) and also keep the interactive HTML. The plot must show base value, final output, feature names and values and be readable.
3. Fill the feature contribution table: top 3 features with patient value, SHAP value, direction, plus "remaining features combined".
4. Write the interpretation: base value → final output, largest upward/downward contributions, whether it is plausible, what a false positive/negative would mean. Describe a model prediction, not a diagnosis.

**Done when:** `outputs/task3/` has the PNG (+HTML), and the contribution table and text are drafted.

---

## Phase 4: Task 4, Global explanation (summary plot)

1. Compute SHAP values for the held-out test set (the whole test set is fine at this size; if you sample, record the rule and seed).
2. Produce the **summary (beeswarm) plot** for the stroke = 1 output. Make labels readable and include the colour legend. A mean |SHAP| bar chart may be added alongside, but only the summary plot is required.
3. Report the number of displayed features and the positive-class count in the explained sample.
4. Feature representation: use the encoded columns as they are (simplest). State this. Do not group columns unless you apply the template's grouping rule consistently.
5. Fill the top-5 table (rank, feature, mean |SHAP|, observed pattern). Read the colour legend correctly; do not treat one-hot code size as clinical severity.
6. Write: most influential features, variation across patients, direction of effects, how the Task 3 case compares with the global pattern, and a note on correlated predictors sharing attribution.

**Done when:** summary plot PNG and top-5 table are saved and the global text is drafted.

---

## Phase 5: Task 6, Different black-box model (do this before the slides so the report is complete)

1. Pick **one** genuinely different model type from Task 2 (e.g. `SVC(probability=True)` or an `MLPClassifier`). Use the **same split, same preprocessing, same predictors**, and the same training oversampling.
2. Explain it with **one** route only:
   - **Kernel SHAP** (`shap.KernelExplainer`) with a prediction function returning stroke = 1 probability, a small training-sample background (e.g. `shap.sample` or kmeans of ~50-100 rows, record size and seed), and a limited number of explained test rows (Kernel SHAP is slow; record how many, and `nsamples`).
   - or **Deep SHAP** only if you use a compatible neural network framework that is already installed.
   Do not run both.
3. Record: model, explainer class, output/class index, scale, background size and seed, explained-sample size, runtime, warnings, and the reconstruction residual (base + sum SHAP vs model output).
4. Evaluate the model on the same test set with the same metrics as Task 2.
5. Produce **Figure 4** (local explanation for the **same case** as Task 3) and **Figure 5** (global summary plot for the explained sample). State the output units in both.
6. Fill the template's Task 2 vs Task 6 comparison table. Note that SHAP magnitudes are only comparable if both are on the same output scale (both probability here); if the explained samples differ, say so.
7. Write: main contributions for this model and whether they agree with the Task 2 model.

**Done when:** metrics, Figures 4 and 5, residual check and comparison table are saved.

---

## Phase 6: Report (fill in the template)

Open `Stroke_SHAP_Report_Template.docx` and produce `Stroke_SHAP_Report_Completed.docx`. **Keep the template's structure, headings and tables.** Do not add sections that are not in the template.

1. Replace **every** `[bracket placeholder]` with real content from your runs. Remove the instructional "How to use this template" text and any grey instruction notes.
2. Insert figures in the marked positions with captions: Figure 1 (EDA), 2 (force plot), 3 (summary plot), 4 (Task 6 local), 5 (Task 6 global).
3. Fill all tables: attribute table (keep), data-quality table, test metrics table, verification table, local contribution table, top-5 table, Task 6 route table (delete the unused column or mark it not used), and the comparison table.
4. Write the Abstract (150-200 words) **last**, and the Conclusion (100-150 words). Each claim must be backed by a reported result.
5. Section 8 (comparison/limitations): cover provenance, representativeness, class imbalance, missingness, no prospective validation, whether calibration was assessed (say plainly if it was **not**), correlated features and background selection. A repeat with a second background sample is only "if feasible", so skip it unless it is quick, and report honestly either way.
6. Student name/ID/course/lecturer/date, repo URL and commit, work contributions and tool-use disclosure: **leave these clearly marked for the student to fill in**. Do not invent personal details. In the tool-use disclosure, state that Claude Code was used to run the analysis, if the course policy requires disclosure (the student decides the final wording).
7. References: cite the Kaggle dataset (fedesoriano, Stroke Prediction Dataset), the SHAP paper (Lundberg & Lee, 2017) and the WHO source only if you repeat the WHO claim (otherwise omit the WHO statistic rather than guessing a year).
8. Complete the reproducibility appendix (versions, seeds, split sizes, parameters, explainer settings, list of evidence files with paths) and tick the final checklist only for items that are truly done.

**Done when:** no `[` placeholders remain except the student-personal fields, and all figures/tables are in.

---

## Phase 7: Task 5, Three slides

Create **exactly three slides**, one per audience, all using the **same model, same case and the same numbers** as the report.

1. **Slide 1: Data scientist.** Model and split, positive class, main metrics (not accuracy alone), top features from the summary plot, explainer and background choice, one limitation (e.g. imbalance, correlated features, leakage risk).
2. **Slide 2: Doctor.** The Task 3 force plot for the same case. Prediction, main features pushing up/down, what missed cases (false negatives) mean, need for calibration and external validation. State that SHAP explains model behaviour and is not a diagnosis.
3. **Slide 3: Patient.** Plain words, no feature codes. Address the patient directly. What the model looked at, which factors pushed this example up or down, that the prediction can be wrong, and that a clinician reads it with other information. No blame, no causal claims, no medication advice, and no "your risk is X%".

**Style rules (important: must not look AI-generated or "professional template"):**
- Plain white (or very light) background. One simple font (e.g. Calibri or Arial). Black or dark grey text, maybe **one** accent colour.
- No gradients, no icons, no decorative shapes, no stock images, no emoji, no coloured banner bars, no animations or transitions.
- A short plain title at the top of each slide. Title in sentence case, no clever taglines.
- Mostly the plot plus a handful of short bullet points written the way a student would write them. Avoid buzzwords and polished marketing phrasing. Avoid em dashes, "leverage", "robust", "delve", "seamless", "key takeaways", "at a glance", and similar.
- No more than about 4-5 short bullets per slide. Text readable at presentation size.
- Do not add title slides, agenda slides, thank-you slides or extra slides. Exactly three.
- Use the real saved plot images from `outputs/`. Add a small source note (e.g. "Dataset: Kaggle stroke prediction, fedesoriano").
- Check all three slides agree with the report (case, predicted value, output units, top features).

Save as `outputs/slides/Stroke_SHAP_Slides.pptx` (use the pptx skill if available, otherwise `python-pptx` if installed). Render to images and check them visually for overflow or unreadable text.

**Done when:** a 3-slide .pptx exists, checked visually, consistent with the report.

---

## Phase 8: Final verification (the task list's last step)

1. Re-run the full pipeline from scratch (`work/` scripts) and confirm results match what is in the report (metrics, case, top features). Fix any mismatch in the **report**, not by editing numbers by hand without re-running.
2. Open the final .docx and check: no leftover placeholders, figures present and readable, captions present, tables filled, headings intact.
3. Open the slides and check: exactly 3 slides, text fits, consistent with the report.
4. Confirm there is no data leakage: split before imputation/encoding/oversampling, oversampling on train only, test set untouched, `id` and `stroke` not in predictors.
5. Confirm no unsupported claims: no causal wording, no "future risk" wording, no diagnosis or treatment advice.
6. Run through the template's "Final submission check" list and report which items are complete.

## Final deliverables (send to the user)

1. `Stroke_SHAP_Report_Completed.docx`
2. `outputs/slides/Stroke_SHAP_Slides.pptx`
3. The `work/` scripts, `outputs/` evidence files and `requirements.txt`

Finish with a short summary: what was produced, the headline results (model metrics, top features), anything that failed or was approximated, and the fields the student still needs to fill in personally (name, ID, course, lecturer, date, repo URL/commit, contributions, tool disclosure).