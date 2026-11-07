# Progress log

## Phase 0: Setup and checks (2026-11-07) - BLOCKED, partly done
- CSV loads: 5110 rows, 12 columns. Dtypes as expected. bmi has 201 missing values. stroke: 4861 zero, 249 one.
- Real spelling in the CSV is `Govt_job` (template says `Govt_jov`).
- Template opens as a valid docx and its placeholders were listed.
- Python 3.13.0. Installed: scikit-learn 1.8.0, pandas 2.2.3, numpy 2.1.2, matplotlib 3.10.8, seaborn 0.13.2, torch 2.14.1.
- MISSING: shap, imbalanced-learn (imblearn), python-docx, python-pptx. Also absent from Python 3.11. No venv or conda env found.
- Waiting for the user's decision, because the plan says not to install packages.
