# Stroke prediction with SHAP (WQF7009, Assignment 1)

Mohaned Altater. Explainable AI assignment: exploratory data analysis, a RandomForest explained with SHAP
(TreeExplainer), one local force plot, one global summary plot, three audience slides, and a second black-box
model (MLP) explained with Kernel SHAP.

## Where to look

| What | Where |
|---|---|
| **Report** (main deliverable) | [`Stroke_SHAP_Report_Completed.docx`](Stroke_SHAP_Report_Completed.docx) |
| **Three slides** (data scientist, doctor, patient) | [`outputs/slides/Stroke_SHAP_Slides.pptx`](outputs/slides/Stroke_SHAP_Slides.pptx) |
| Code written for this assignment | [`work/`](work/) |
| All results, figures and tables used in the report | [`outputs/`](outputs/) |
| Exact package versions | [`requirements.txt`](requirements.txt) |
| Step-by-step log of what was done | [`outputs/progress_log.md`](outputs/progress_log.md) |

Every number in the report and slides is read from the files in `outputs/`, which are produced by the scripts in `work/`.

## Scripts in `work/` (run in this order by `run_all.py`)

| Script | Task | Output folder |
|---|---|---|
| `common.py` | Shared pipeline: stratified 80/20 split first, then bmi median and one-hot encoder fitted on training data only, oversampling of the training set only | (used by all) |
| `01_eda.py` | Task 1: data-quality facts and Figure 1 | `outputs/eda/` |
| `02a_select_threshold.py` | Decision thresholds from 5-fold out-of-fold training predictions (max F2) | `outputs/threshold/` |
| `02_train_shap.py` | Task 2: RandomForest, test metrics, TreeExplainer, additivity check | `outputs/task2/` |
| `03_task3_local.py` | Task 3: force plot for one held-out case | `outputs/task3/` |
| `04_task4_global.py` | Task 4: summary (beeswarm) plot and top-5 table | `outputs/task4/` |
| `05_task6.py` | Task 6: MLP explained with Kernel SHAP, comparison with Task 2 | `outputs/task6/` |
| `06_background_repeat.py` | Kernel SHAP repeated with a second background sample | `outputs/task6/` |
| `08_build_slides.py` | Task 5: the three slides | `outputs/slides/` |
| `07_build_report.py` | Fills the report template from `outputs/` | report `.docx` |
| `09_final_checks.py` | Leakage, consistency and wording checks | `outputs/final_checks.json` |

## Course files

The course code was downloaded as a ZIP and is kept unchanged.

- `00_data_exploration.py` and `utils.py`: Task 1 runs `00_data_exploration.py` exactly as supplied.
- `03_shap.py`: the course starting point for Tasks 2 to 4. Its preprocessing was not reused (see the report, Section 3.1).
- `Stroke_SHAP_Report_Template.docx`: the blank report template.
- `course_code_unused/`: course scripts this assignment did not use (interpretable models, LIME, counterfactuals, LRP). They import `utils.py`, so run them from the top level if ever needed.

## How to reproduce

1. Download `healthcare-dataset-stroke-data.csv` from the Kaggle Stroke Prediction Dataset by fedesoriano
   (https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and put it in `data/`.
   The file is not included here because its licence is "Data files © Original Authors".
   The expected SHA-256 is `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`.
2. With Python 3.13:
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   .venv\Scripts\python work\run_all.py
   ```
   This rebuilds everything in `outputs/`, the slides and the report, then runs the final checks.
   Kernel SHAP and the threshold cross-validation take a few minutes in total.

## Tool use

Claude Code (Anthropic) was used as an AI coding assistant under my direction. The tool-use disclosure is in the
report, Section 8.4.
