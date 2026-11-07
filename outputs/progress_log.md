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
