# Progress log

## Phase 0: Setup and checks (2026-11-07) - done
- CSV loads: 5110 rows, 12 columns. bmi has 201 missing values. stroke: 4861 zero, 249 one.
- Real spelling in the CSV is `Govt_job` (template says `Govt_jov`).
- Template opens; placeholder list saved in outputs/template_placeholders.txt.
- PROBLEM: shap, imbalanced-learn, python-docx and python-pptx were not installed anywhere. The plan says not to install, so I asked the user. Network was on and the user approved installing them in a project venv (.venv, git-ignored). System Python was not changed.
- Versions used are in requirements.txt (pip freeze of the venv). The venv resolved newer versions than system Python (scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, matplotlib 3.11.2, shap 0.54.1).
- All imports OK with .venv/Scripts/python.
