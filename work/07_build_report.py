"""Phase 6: fill the report template from the saved evidence files.

Numbers come from outputs/ so the report always matches the last pipeline run.
Personal fields stay as highlighted [Student: ...] markers.
"""
import copy
import json
import re

import docx
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Inches

from common import OUT, ROOT

TEMPLATE = ROOT / "Stroke_SHAP_Report_Template.docx"
TARGET = ROOT / "Stroke_SHAP_Report_Completed.docx"


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


eda = load("eda/eda_facts.json")
split = load("task2/split_info.json")
m2 = load("task2/metrics.json")
ver = load("task2/verification.json")
case = load("task3/selected_case.json")
glob = load("task4/global_summary.json")
m6 = load("task6/metrics.json")
rec = load("task6/explainer_record.json")
comp = load("task6/comparison.json")
rep = load("task6/background_repeat.json")
thr = load("threshold/thresholds.json")
t_rf, t_mlp = thr["models"]["random_forest"], thr["models"]["mlp"]
m2h, m6h = m2["metrics_at_0_5"], m6["metrics_at_0_5"]
assert abs(m2["threshold"] - t_rf["threshold"]) < 1e-12 and abs(m6["threshold"] - t_mlp["threshold"]) < 1e-12
INFO_PATH = ROOT / "work" / "student_info.json"
info = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {}


def student(key, marker):
    """Real value from work/student_info.json if given, else a highlighted marker."""
    value = str(info.get(key) or "").strip()
    return value if value else f"[Student: {marker}]"
num = pd.read_csv(OUT / "eda/numeric_summary.csv").set_index("variable")
cats = pd.read_csv(OUT / "eda/categorical_summary.csv")
miss = pd.read_csv(OUT / "eda/missing_values.csv").set_index("column")
t4_table = pd.read_csv(OUT / "task4/mean_abs_shap_all_features.csv").set_index("feature")
reqs = dict(line.strip().split("==") for line in (ROOT / "requirements.txt").read_text().splitlines()
            if "==" in line)


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def signed(x): return f"{x:+.3f}"
def pct(x): return f"{x * 100:.2f}%"
def tf(x): return f"{x:.3g}"


def versions():
    keys = ["shap", "scikit-learn", "imbalanced-learn", "pandas", "numpy", "matplotlib"]
    return ", ".join(f"{k} {reqs[k]}" for k in keys)


d = docx.Document(TEMPLATE)
HIGHLIGHT = re.compile(r"(\[Student:[^\]]*\])")


def set_runs(par, text):
    """Replace a paragraph's text, keeping size and font but dropping grey italics."""
    rpr = None
    if par.runs and par.runs[0]._r.rPr is not None:
        rpr = copy.deepcopy(par.runs[0]._r.rPr)
        for tag in ("w:i", "w:iCs", "w:color"):
            for el in rpr.findall(qn(tag)):
                rpr.remove(el)
    for child in list(par._p):
        if child.tag != qn("w:pPr"):
            par._p.remove(child)
    for piece in HIGHLIGHT.split(text):
        if not piece:
            continue
        run = par.add_run(piece)
        if rpr is not None:
            if run._r.rPr is not None:
                run._r.remove(run._r.rPr)
            run._r.insert(0, copy.deepcopy(rpr))
        if HIGHLIGHT.fullmatch(piece):
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def find(prefix):
    hits = [p for p in d.paragraphs if p.text.strip().startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} paragraphs start with {prefix!r}")
    return hits[0]


def fill(prefix, text):
    set_runs(find(prefix), text)


def fill_many(prefix, texts):
    """First text replaces the paragraph; the others follow as new paragraphs."""
    par = find(prefix)
    set_runs(par, texts[0])
    anchor = par
    for text in texts[1:]:
        new = copy.deepcopy(par._p)
        anchor._p.addnext(new)
        anchor = docx.text.paragraph.Paragraph(new, par._parent)
        set_runs(anchor, text)


def drop(prefix):
    p = find(prefix)._p
    p.getparent().remove(p)


def picture(prefix, path, width):
    par = find(prefix)
    for child in list(par._p):
        if child.tag != qn("w:pPr"):
            par._p.remove(child)
    par.add_run().add_picture(str(OUT / path), width=Inches(width))
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER


def cell(table, r, c, text):
    target = d.tables[table].rows[r].cells[c]
    pars = target.paragraphs
    for extra in pars[1:]:
        extra._p.getparent().remove(extra._p)
    set_runs(pars[0], text)


# ---- Front matter ----
fill("Assignment report template", "Assignment report")
fill("Student name:", "Student name: " + student("name", "enter name"))
fill("Student ID:", "Student ID: " + student("student_id", "enter ID"))
fill("Course and programme:", "Course and programme: " + student("course", "enter details"))
fill("Lecturer:", "Lecturer: " + student("lecturer", "enter name"))
fill("Submission date:", "Submission date: " + student("date", "enter date"))
fill("Repository URL and commit:", "Repository URL and commit: " + student("repository", "enter URL and commit"))
drop("How to use this template")
drop("Replace every square bracket")
drop("The GitHub repository, dataset file")
for par in d.sections[0].header.paragraphs:
    for run in par.runs:
        run.text = run.text.replace("REPORT TEMPLATE", "REPORT")
d.core_properties.title = "Stroke Classification and SHAP Explanation"

abstract = (
    f"This report classifies recorded stroke status in the Kaggle stroke prediction dataset and explains "
    f"the predictions with SHAP. The data hold {eda['rows']} patients, of whom {eda['stroke_1']} "
    f"({eda['stroke_pct_positive']}%) have stroke = 1. After a stratified 80/20 split, bmi was imputed with "
    f"the training median, categories were one-hot encoded on training data, and only the training set was "
    f"oversampled. Decision thresholds were chosen by maximising F2 on five-fold out-of-fold training predictions. "
    f"The Task 2 RandomForest, explained with TreeExplainer, reached ROC AUC {f3(m2['roc_auc'])} and average "
    f"precision {f3(m2['average_precision'])}; at its threshold of {tf(m2['threshold'])} it found {m2['tp']} of "
    f"{m2['positive_support']} test strokes (recall {f3(m2['recall_stroke1'])}) with precision "
    f"{f3(m2['precision_stroke1'])} and {m2['fp']} false alarms, against recall {f3(m2h['recall_stroke1'])} at 0.5. "
    f"For the selected held-out true positive, the predicted probability rose from a base value of "
    f"{f3(case['base_value'])} to {f2(case['predicted_probability'])}, mainly through heart disease and glucose, while "
    f"bmi pushed it down. Globally, age, average glucose and bmi had the largest mean absolute SHAP values. An MLP "
    f"explained with Kernel SHAP ranked cases less well (ROC AUC {f3(m6['roc_auc'])}, recall "
    f"{f3(m6['recall_stroke1'])} at its threshold) and agreed on age as the top feature while differing on several "
    f"others. Scores are uncalibrated and the data retrospective, so this explains model behaviour, not clinical use."
)
assert 150 <= len(abstract.split()) <= 200, len(abstract.split())
fill("[Write approximately 150", abstract)

# ---- Section 1 ----
fill("[Introduce the stroke classification",
     "Stroke is a serious medical event, and a model that flags patients with a recorded stroke from routine "
     "health data is only useful if people can see what drives its output. A score alone gives a clinician "
     "little to question. SHAP [2] breaks one prediction into feature contributions, which makes it possible to "
     "check whether a model relies on sensible information. The WHO statistic in the brief is not repeated here "
     "because its reference year could not be verified from the supplied files.")
fill("Objective:",
     "Objective: Evaluate a classifier for stroke = 1 on an untouched held-out test set, using metrics that suit "
     "a rare class, and explain its predictions with SHAP at the level of one case and of the whole test set. A "
     "second, different model is then explained in the same way for comparison.")
fill("Outcome definition:",
     "Outcome definition: stroke = 1 indicates a recorded history of stroke; stroke = 0 indicates no recorded "
     "stroke. The supplied files only link to the dataset page and give no follow-up period, so the output is "
     "not described as a validated future stroke risk anywhere in this report.")
fill("Dataset URL and access date:",
     f"Dataset URL and access date: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset [1]; the CSV was "
     f"supplied with the course code, so no separate download date was recorded. File name and version: "
     f"healthcare-dataset-stroke-data.csv, SHA-256 {eda['file_sha256'][:16]}... (full value in Section 9.2). "
     f"Population, collection period and provenance: not stated in the supplied files, so treated as unavailable.")
cell(1, 3, 1, "Patient age. Treated as years; range 0.08 to 82, with fractional values for infants")
cell(1, 7, 1, "children, Govt_job, Never_worked, Private, Self-employed")
cell(1, 9, 1, "Average blood glucose; unit not stated in the supplied files")
cell(1, 10, 1, "Body mass index; unit not stated in the supplied files (kg/m² by the usual definition of BMI)")
fill("Data checks:",
     "Data checks: All 12 column names were confirmed in the CSV. The real spelling is Govt_job, not Govt_jov as "
     "written in the brief. Unknown smoking status means unavailable information, so it was kept as its own "
     "category and not merged with never smoked.")

# ---- Section 2 ----
fill("GitHub download method:",
     f"GitHub download method: {student('code_source', 'confirm clone or ZIP; the folder name xai-series-master suggests a ZIP of the master branch')}. "
     f"Commit or release: {student('code_commit', 'enter if known; no commit is recorded in the supplied files')}. Python environment and dependencies: Python {ver['python']} on Windows 11 in a project virtual "
     f"environment; {versions()}. Full list in requirements.txt.")
fill("Execution command:",
     "Execution command: python 00_data_exploration.py, no arguments, run from the project root. MPLBACKEND=Agg was "
     "set in the environment so that plt.show() would not wait for a window. Date, output directory and execution "
     "status: 2026-11-07, console output saved to outputs/eda/00_data_exploration_stdout.txt, exit code 0. "
     "Modifications or errors resolved: none; the script ran unchanged and no errors occurred.")

cell(2, 1, 1, f"{eda['rows']} rows; {eda['columns']} columns")
cell(2, 1, 2, "Imported dimensions confirmed. After removing id and stroke, 10 original predictors remain")
cell(2, 2, 1, f"{eda['duplicate_rows']} duplicate rows; {eda['duplicate_ids']} duplicate ids")
cell(2, 2, 2, "Nothing to remove. Also no duplicates when id is ignored")
cell(2, 3, 1, f"bmi: {int(miss.loc['bmi', 'missing_n'])} ({miss.loc['bmi', 'missing_pct']}%) blank. "
              f"smoking_status Unknown: {int(miss.loc['smoking_status (Unknown)', 'missing_n'])} "
              f"({miss.loc['smoking_status (Unknown)', 'missing_pct']}%). All other columns: 0")
cell(2, 3, 2, f"bmi imputed with the training median. Blank bmi is uneven: "
              f"{eda['stroke_rate_bmi_missing_pct']}% of those rows have stroke = 1 vs "
              f"{eda['stroke_rate_bmi_present_pct']}% with bmi recorded. Unknown smoking kept as a category")
cell(2, 4, 1, f"{eda['stroke_0']}; {eda['stroke_1']}; {eda['stroke_pct_positive']}% positive")
cell(2, 4, 2, "Strong imbalance. Always predicting 0 would give about 95% accuracy, so accuracy alone is not "
              "reported. Training set oversampled")


def num_line(var):
    r = num.loc[var]
    return (f"{var}: {r['mean']} ± {r['sd']}; median {r['median']}, IQR {r['iqr']} "
            f"({r['q1']} to {r['q3']}); range {r['min']} to {r['max']}")


cell(2, 5, 1, "; ".join([num_line("age"), num_line("avg_glucose_level"), num_line("bmi")]).replace("; age", "age"))
cell(2, 5, 2, f"{eda['age_under_1']} ages under 1 year and {eda['age_under_18']} under 18. bmi extremes: "
              f"{eda['bmi_under_12']} rows under 12, {eda['bmi_over_60']} over 60. Unusual values were kept and flagged")
cat_text = []
for var, group in cats[~cats.variable.isin(["hypertension", "heart_disease"])].groupby("variable", sort=False):
    cat_text.append(var + ": " + ", ".join(f"{r.level} {r.n} ({r.pct}%)" for r in group.itertuples()))
for var in ["hypertension", "heart_disease"]:
    group = cats[cats.variable == var]
    cat_text.append(var + ": " + ", ".join(f"{r.level}: {r.n} ({r.pct}%)" for r in group.itertuples()))
cell(2, 6, 1, "; ".join(cat_text))
cell(2, 6, 2, f"Rare levels: gender Other ({eda['gender_other']} row) and Never_worked. These have too few rows for "
              f"any conclusion. Categories are one-hot encoded and the code size has no clinical meaning")

picture("[Insert eda panel here", "eda/figure1_eda_panel.png", 6.3)
captions = [p for p in d.paragraphs if p.text.strip().startswith("Caption:")]
assert len(captions) == 3
set_runs(captions[0],
         f"Figure 1. EDA panel for all {eda['rows']} rows, before any modelling. A and B show patient counts; C to E "
         f"show the percentage within each stroke class; F and G show box plots by class; H shows the share of "
         f"patients with stroke = 1 by condition. Stroke = 1 rows are rare and tend to be older, as an association.")
fill("[Summarise three evidence-based",
     f"Three observations follow. First, the classes are imbalanced: {eda['stroke_1']} of {eda['rows']} rows "
     f"({eda['stroke_pct_positive']}%) have stroke = 1 (Figure 1A). Second, information is missing unevenly: bmi is "
     f"blank in {int(miss.loc['bmi', 'missing_n'])} rows and smoking is Unknown in "
     f"{int(miss.loc['smoking_status (Unknown)', 'missing_n'])} (Figure 1B), and rows with blank bmi have a stroke "
     f"share of {eda['stroke_rate_bmi_missing_pct']}% against {eda['stroke_rate_bmi_present_pct']}% for the rest. "
     f"Third, age is associated with stroke status: median age is {eda['median_age_by_stroke']['1']:.0f} with "
     f"stroke = 1 and {eda['median_age_by_stroke']['0']:.0f} without (Figure 1C and 1F). Hypertension and heart "
     f"disease show a similar association (Figure 1H). These are associations in a retrospective dataset and do "
     f"not show causes. They shaped the plan: test metrics focus on the positive class, the training set is "
     f"oversampled, bmi is imputed from training data only, and unusual values are kept because no rule in the "
     f"source says they are errors.")
fill("Training safeguards:",
     "Training safeguards: The data were split before imputation, encoding or resampling. Each of these steps was "
     "fitted on training data only, and the test set was left untouched apart from applying the fitted steps. id and "
     "stroke are excluded from the predictors. The script stops if these checks fail.")

# ---- Section 3 ----
fill("For an additive explanation",
     "For an additive explanation, the base value plus the sum of feature contributions reconstructs the explained "
     "output on its stated scale. Here the output is already a probability, so no link function or transformation "
     "applies.")
fill("Task 2 classifier and rationale:",
     f"Task 2 classifier and rationale: RandomForestClassifier with random_state=2021 and default hyperparameters, "
     f"as in the course script. A tree ensemble handles mixed binary and numeric inputs without scaling and can be "
     f"explained exactly by TreeExplainer. Predictors retained and excluded: all columns except id and stroke. "
     f"Preprocessing pipeline: bmi imputed with the training median ({split['bmi_train_median']}); gender, "
     f"ever_married, work_type, Residence_type and smoking_status one-hot encoded with an encoder fitted on training "
     f"data; no scaling. This gives {split['n_features']} encoded features, for example ever_married_Yes and "
     f"smoking_status_Unknown. The course utils.py was not used for preprocessing because it fills bmi with 0 and "
     f"preprocesses before the split.")
fill("Train validation test sizes:",
     f"Train validation test sizes: {split['n_train']} train (80%, {split['train_pos']} positive) and "
     f"{split['n_test']} test (20%, {split['test_pos']} positive); no separate validation set, because the threshold "
     f"was chosen from cross-validated predictions inside the training split. Stratification, seed "
     f"and any group separation: stratified on stroke, random_state=2021; no group separation was needed because all "
     f"ids are unique. Hyperparameter tuning and imbalance strategy: no tuning; RandomOverSampler on the training "
     f"set only, giving {split['n_train_oversampled']} rows with {split['train_oversampled_pos']} positive. Decision "
     f"threshold and selection rule: chosen on the training split only, before any test result was looked at. "
     f"{thr['n_splits']}-fold stratified cross-validation (shuffled, seed {thr['cv_seed']}) was run on the "
     f"{thr['n_rows']} un-oversampled training rows. In each fold the bmi median and encoder were fitted on the "
     f"fold's training rows, only those rows were oversampled, and the held-out fold was scored as it was. The "
     f"threshold maximises F2 for stroke = 1 on the out-of-fold predictions, with candidates from "
     f"precision_recall_curve and ties going to the higher threshold. F2 weights recall twice as much as precision, "
     f"because a missed stroke is the more serious error here. RandomForest threshold {tf(t_rf['threshold'])}: "
     f"out-of-fold recall {f3(t_rf['oof_recall'])}, precision {f3(t_rf['oof_precision'])}, "
     f"{t_rf['oof_share_flagged'] * 100:.1f}% of training rows flagged. MLP threshold {tf(t_mlp['threshold'])}: recall "
     f"{f3(t_mlp['oof_recall'])}, precision {f3(t_mlp['oof_precision'])}, {t_mlp['oof_share_flagged'] * 100:.1f}% "
     f"flagged. Both final models were then evaluated once on the test set at these thresholds.")
cell(3, 1, 1, f"{f3(m2['accuracy'])} and {f3(m2['balanced_accuracy'])}")
cell(3, 1, 2, f"Threshold {tf(m2['threshold'])}, chosen on training folds (Section 3.1)")
cell(3, 2, 1, f"Precision {f3(m2['precision_stroke1'])}, recall {f3(m2['recall_stroke1'])}, F1 {f3(m2['f1_stroke1'])}")
cell(3, 2, 2, f"Positive class support: {m2['positive_support']} of {split['n_test']} test rows")
cell(3, 3, 1, f"{f3(m2['roc_auc'])} and {f3(m2['average_precision'])}")
cell(3, 3, 2, f"Both use predicted probabilities. PR metric is average precision; test prevalence is "
              f"{pct(m2['prevalence_test'])}")
cell(3, 4, 1, f"TN {m2['tn']}, FP {m2['fp']}, FN {m2['fn']}, TP {m2['tp']}")
cell(3, 4, 2, f"The model missed {m2['fn']} of {m2['positive_support']} recorded strokes but raised {m2['fp']} false "
              f"alarms. At 0.5 it found only {m2h['tp']} of {m2h['positive_support']} (recall "
              f"{f3(m2h['recall_stroke1'])}, precision {f3(m2h['precision_stroke1'])})")

fill("Explainer class and suitability:",
     f"Explainer class and suitability: shap.TreeExplainer, a compatible exact explainer for tree ensembles. SHAP "
     f"version and model library version: shap {ver['shap_version']}, scikit-learn {ver['sklearn_version']}. The "
     f"explainer works on the fitted RandomForest and takes the transformed input directly: the {split['n_features']} "
     f"encoded columns after imputation, in the order given in Section 9.2.")
fill("Output explained:",
     f"Output explained: predicted probability of stroke = 1, with no link function. Class label and output index: "
     f"model.classes_ is {ver['model_classes']}, so stroke = 1 is index {ver['stroke1_class_index']}. Installed shap "
     f"returns one array of shape {tuple(ver['shap_values_raw_shape'])} (rows, features, classes), not the list "
     f"assumed by 03_shap.py, so the class was selected on the last axis. Background source, size and sampling seed: "
     f"none passed; feature_perturbation is tree_path_dependent, which uses training coverage stored in the trees. "
     f"Explanation sample source and size: all {ver['n_explained']} held-out test rows.")
cell(4, 1, 1, f"Model feature_names_in_ equals the test columns in the same order: "
              f"{ver['feature_names_match_model']}. Test matrix {split['n_test']} × {split['n_features']}; SHAP array "
              f"{tuple(ver['shap_values_raw_shape'])}")
cell(4, 2, 1, f"classes_ = {ver['model_classes']}; index {ver['stroke1_class_index']} is stroke = 1. Units: predicted "
              f"probability, model_output = {ver['explainer_model_output']}")
cell(4, 3, 1, f"Row 0 of the test set: {f4(ver['base_value'])} + sum of SHAP = {f4(ver['example_row0']['base_plus_sum'])}")
cell(4, 4, 1, f"Row 0 model output {f4(ver['example_row0']['model_output'])}. Over all {ver['n_explained']} rows, max "
              f"absolute residual {ver['max_abs_residual']:.2e}, tolerance {ver['tolerance']:.0e}: passed")
fill("[Explain what positive and negative SHAP values",
     f"A positive SHAP value means that the feature value moved this model's predicted probability of stroke = 1 "
     f"above the base value; a negative value moved it below. The base value is {f3(ver['base_value'])}, not close "
     f"to the test prevalence of {pct(m2['prevalence_test'])}, because the trees were fitted on balanced oversampled "
     f"data. Most test rows therefore have many negative contributions. SHAP describes this model relative to that "
     f"reference; it does not establish that a feature causes stroke.")

# ---- Section 4 ----
rv = case["raw_values"]
fill("Anonymous row reference:",
     f"Anonymous row reference: test_row {case['test_row']} (position in the held-out test set, not id). Selection "
     f"rule: the true-positive test case with the highest predicted probability. This is a favourable choice: it is "
     f"the highest-scoring of {case['n_true_positives_available']} true positives, so it shows the model at its "
     f"clearest and does not represent the {m2['fn']} missed cases or the {m2['fp']} false alarms. Actual class: "
     f"{case['actual_class']}. Predicted stroke class: {case['predicted_class']}. Threshold: {tf(case['threshold'])}.")
fill("Model score or probability:",
     f"Model score or probability: {f2(case['predicted_probability'])}. Explained output and units: predicted "
     f"probability of stroke = 1. Base value in those units: {f3(case['base_value'])}. Case type: true positive.")
picture("[Insert local shap force plot here", "task3/force_plot_case.png", 6.3)
set_runs(captions[1],
         f"Figure 2. RandomForest force plot, test_row {case['test_row']}: base value {f3(case['base_value'])} to "
         f"output {f2(case['predicted_probability'])}, predicted probability of stroke = 1. Red pushes up, blue down.")


def pretty_value(row):
    return f"{row['encoded_value']:g}"


for i, row in enumerate(case["top3"]):
    cell(5, i + 1, 0, row["feature"])
    cell(5, i + 1, 1, pretty_value(row))
    cell(5, i + 1, 2, signed(row["shap"]))
    cell(5, i + 1, 3, "Increases" if row["shap"] > 0 else "Decreases")
rest = case["remaining_features_sum"]
cell(5, 4, 2, signed(rest))
cell(5, 4, 3, "Net increase" if rest > 0 else "Net decrease")
up1, up2 = case["largest_up"][:2]
dn = case["largest_down"][0]
fill("[Starting from the base value",
     f"Starting from the base value of {f3(case['base_value'])}, the model output moved to "
     f"{f2(case['predicted_probability'])} on the probability scale. The largest upward contributions were "
     f"{up1['feature']} = {pretty_value(up1)} ({signed(up1['shap'])}) and {up2['feature']} = {pretty_value(up2)} "
     f"({signed(up2['shap'])}). The largest downward contribution was {dn['feature']} = {pretty_value(dn)} "
     f"({signed(dn['shap'])}). The other {split['n_features'] - 3} features together added {signed(rest)}. These contributions add up to {signed(case['sum_shap'])}, with a "
     f"residual of {case['residual']:.1e}.")
fill("[Discuss whether the explanation is plausible",
     f"The upward pushes are plausible as model behaviour, because heart disease and high glucose are also linked "
     f"with stroke = 1 in Figure 1. The downward push from bmi = {rv['bmi']} needs care. It should not be read as "
     f"higher bmi being protective: Section 5 shows that bmi has no simple direction in this model. Age {rv['age']:g} "
     f"contributes only {signed(glob['case_age_shap'])} here even though it is the top feature globally. No data "
     f"were missing for this case. A false positive "
     f"would flag someone with no recorded stroke and could lead to needless worry or checks. A false negative would "
     f"leave a recorded stroke unflagged. At the chosen threshold both happen: {m2['fn']} of {m2['positive_support']} "
     f"recorded strokes were missed and {m2['fp']} rows without one were flagged. "
     f"This is a model prediction on retrospective data, not a diagnosis or a treatment recommendation.")

# ---- Section 5 ----
fill("Model and output explained:",
     f"Model and output explained: Task 2 RandomForest, predicted probability of stroke = 1. Held-out explanation "
     f"sample and positive class count: all {glob['n_explained']} test rows, {glob['n_positive']} with stroke = 1. "
     f"Number of displayed features: {glob['n_features_displayed']} of {glob['n_features_total']}. Sampling rule and "
     f"seed: no sampling, so no seed applies.")
fill("Feature representation:",
     "Feature representation: encoded columns as they are, with no grouping. Each category of a one-hot variable has "
     "its own row, so a single original variable can appear more than once.")
picture("[Insert global shap summary plot here", "task4/summary_plot.png", 5.6)
set_runs(captions[2],
         f"Figure 3. SHAP summary plot for RandomForest on all {glob['n_explained']} held-out rows "
         f"({glob['n_positive']} with stroke = 1). The x axis is the change in predicted probability of stroke = 1. Each "
         f"dot is one patient, coloured from low (blue) to high (red) feature value. Older age tends to push the output "
         f"up, and age has the widest spread.")


def pattern_text(row):
    if row["rule"] == "value 0 vs value 1":
        return (f"Value 1 pushes up (mean {signed(row['mean_shap_high'])}); value 0 pushes slightly down "
                f"(mean {signed(row['mean_shap_low'])})")
    return None


t5 = {r["feature"]: r for r in glob["top5"]}
assert [r["feature"] for r in glob["top5"]] == ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
age, glu, bmi = t5["age"], t5["avg_glucose_level"], t5["bmi"]
top_text = {
    "age": f"Higher values push up. Lowest quarter mean {signed(age['mean_shap_low'])}, highest quarter "
           f"{signed(age['mean_shap_high'])}; Spearman value vs SHAP {age['spearman_value_vs_shap']:+.2f}",
    "avg_glucose_level": f"Weak upward link. Lowest quarter mean {signed(glu['mean_shap_low'])}, highest quarter "
                         f"{signed(glu['mean_shap_high'])}; only the highest values reach above zero",
    "bmi": f"No simple direction. Lowest quarter mean {signed(bmi['mean_shap_low'])}, highest quarter "
           f"{signed(bmi['mean_shap_high'])}; Spearman {bmi['spearman_value_vs_shap']:+.2f}, and middle values sit "
           f"closest to zero",
    "hypertension": pattern_text(t5["hypertension"]),
    "heart_disease": pattern_text(t5["heart_disease"]),
}
for i, row in enumerate(glob["top5"]):
    cell(6, i + 1, 1, row["feature"])
    cell(6, i + 1, 2, f3(row["mean_abs_shap"]))
    cell(6, i + 1, 3, top_text[row["feature"]])
fill("[Identify the most influential",
     f"Age is the most influential feature (mean absolute SHAP {f3(age['mean_abs_shap'])}), followed by "
     f"avg_glucose_level ({f3(glu['mean_abs_shap'])}) and bmi ({f3(bmi['mean_abs_shap'])}). The next two features are "
     f"much smaller. Effects vary across patients: age contributions range from "
     f"{signed(t4_table.loc['age', 'min_shap'])} to {signed(t4_table.loc['age', 'max_shap'])}. Read with the colour "
     f"legend, higher age and a value of 1 for hypertension or heart disease generally increase the output, while "
     f"bmi has no consistent direction. Many SHAP values are negative because most test rows score well below the "
     f"balanced base value of {f3(ver['base_value'])}. For one-hot columns, red only means that the category is "
     f"present; the code size says nothing about clinical severity.")
imp = glob["bmi_imputation_check"]
assert imp["n_positive_bmi_shap_imputed"] > imp["n_positive_bmi_shap"] / 2
assert glob["glucose_positive_shap"]["min_glucose"] > 150
fill_after_identify = (
    f"The bmi result needs a closer look. Only {imp['n_positive_bmi_shap']} test rows have a positive bmi SHAP value, "
    f"all with bmi between {imp['bmi_range_of_positive_shap'][0]} and {imp['bmi_range_of_positive_shap'][1]}, next to "
    f"the imputed value of {imp['imputed_value']}, and {imp['n_positive_bmi_shap_imputed']} of them are imputed rows. "
    f"The {imp['n_imputed_test']} imputed test rows have a mean bmi SHAP of {signed(imp['mean_bmi_shap_imputed'])}, "
    f"against {signed(imp['mean_bmi_shap_not_imputed'])} for the others. The imputation used training data only, so "
    f"this is not leakage from the test set, but the model may be treating the imputed value as a sign that bmi was "
    f"blank, and blank bmi is linked with stroke = 1 in Section 2.")
_identify = find("Age is the most influential")
_new = copy.deepcopy(_identify._p)
_identify._p.addnext(_new)
set_runs(docx.text.paragraph.Paragraph(_new, _identify._parent), fill_after_identify)
rank = {r["feature"]: r for r in glob["case_feature_ranks"]}
assert abs(rank["heart_disease"]["shap"] - t4_table.loc["heart_disease", "max_shap"]) < 1e-12
assert abs(rank["avg_glucose_level"]["shap"] - t4_table.loc["avg_glucose_level", "max_shap"]) < 1e-12
pearson = glob["pearson_corr_test_set"]
fill("[Explain whether the Task 3 case",
     f"The Task 3 case only partly follows the global pattern. Its glucose and bmi contributions come from globally "
     f"important features, but age ranks 1st globally and only {glob['case_age_case_rank']}th for this case, while "
     f"heart_disease ranks 5th globally and 1st for the case. Its heart_disease value of "
     f"{signed(rank['heart_disease']['shap'])} and glucose value of {signed(rank['avg_glucose_level']['shap'])} are "
     f"the largest for those features in the test set, so this is an unusual case and not a typical one. Global "
     f"importance summarises the size of contributions across patients, while a local value gives the direction for "
     f"one person. Correlated predictors can share attribution: age correlates with ever_married_Yes "
     f"(r = {pearson['age_vs_ever_married_Yes']:.2f}) and work_type_children "
     f"(r = {pearson['age_vs_work_type_children']:.2f}) in the test set, and ever_married_Yes and ever_married_No "
     f"hold the same information in two columns. A small value for one of them may not mean that the information is "
     f"unimportant.")

# ---- Section 6 ----
fill("Create exactly three slides",
     "Three slides were made, one per audience, using the same RandomForest, the same case and the same numbers as "
     "this report. The file is outputs/slides/Stroke_SHAP_Slides.pptx.")
fill("Title: [Model performance", "Title: Model performance and explanation of stroke classification")
fill("Visual: [Global summary plot", "Visual: Global summary plot (Figure 3), with the main numbers as short bullets.")
fill("[State the model, split, positive class",
     f"The slide gives the model, the stratified 80/20 split with training-only oversampling, and stroke = 1 as the "
     f"positive class. It reports ROC AUC {f3(m2['roc_auc'])}, average precision {f3(m2['average_precision'])}, recall "
     f"{f3(m2['recall_stroke1'])} and precision {f3(m2['precision_stroke1'])} at the threshold of {tf(m2['threshold'])} "
     f"chosen on training folds, with outputs on the probability scale. "
     f"Top features are age, avg_glucose_level and bmi. The explainer is TreeExplainer with path-dependent settings, "
     f"so no background data were passed. The limitation shown is that the scores come from oversampled training data "
     f"and calibration was not checked.")
fill("[Explain the evidence supporting model reliability",
     "Accuracy is not shown alone. The evidence is one held-out test set with 50 positives, and the additivity check "
     "passed. The threshold came from training folds and was not checked on other data. External validation and a "
     "calibration check would be needed before relying on the model.")
fill("Title: [Interpreting one patient", "Title: Interpreting one patient prediction in clinical context")
fill("Visual: [Force plot", f"Visual: Force plot for the same anonymous case, test_row {case['test_row']} (Figure 2).")
fill("[State the prediction and the main features",
     f"The slide gives the predicted probability of {f2(case['predicted_probability'])} against a base value of "
     f"{f3(case['base_value'])}, with heart disease and glucose pushing up and bmi pushing down. It explains that "
     f"{m2['fn']} of {m2['positive_support']} recorded strokes were still missed at the threshold of "
     f"{tf(m2['threshold'])}, while {m2['fp']} rows without a stroke were flagged, so neither a low nor a high score "
     f"settles anything. It states the need for calibration and external validation, and that SHAP explains "
     f"model behaviour and is not a diagnosis.")
fill("[Explain how a clinician might review",
     "A clinician could use the plot to check whether the drivers look sensible against the patient record. The "
     "dataset is retrospective and educational, so the slide does not suggest any clinical use.")
fill("Title: [What the computer", "Title: What the computer prediction means for this example")
fill("Visual: [A simple labelled display",
     "Visual: A simple chart built from the same SHAP values, showing what pushed the prediction up or down, in words "
     "and without feature codes or numbers.")
fill("[Use first-person patient language",
     "The slide addresses the patient directly. It says that the computer looked at recorded information such as "
     "age, health conditions and blood sugar, names the factors that pushed this example up and down, and says that "
     "the prediction can be wrong. No score is given as a risk percentage.")
fill("[Explain uncertainty in simple language",
     "Uncertainty is explained in simple words: the computer learned from past records, misses some people who had a "
     "stroke and flags many who did not. The "
     "slide makes no causal claims, gives no blame and no medication advice, and says that a clinician would read the "
     "result alongside other information.")
fill("[Confirm the case, predicted value",
     "All three slides use the RandomForest and the same case, predicted value, output units and top features as this "
     "report. A source note credits the Kaggle dataset, and the text was checked at presentation size.")

# ---- Section 7 ----
fill("Task 2 model:",
     f"Task 2 model: RandomForestClassifier. Task 6 model: MLPClassifier, a neural network with one hidden layer of "
     f"{m6['hidden_layer_sizes'][0]} units, default settings and max_iter=500, after a StandardScaler. Reason for "
     f"choosing it: it is a different model type that learns smooth functions, not axis-aligned splits, and its "
     f"predictions are fast enough for Kernel SHAP. The split, encoded predictors, bmi imputation and training "
     f"oversampling are the same as Task 2. The only addition is the scaler, needed by the network and fitted on "
     f"training data as part of the model.")
drop("Suggested optional route:")
cell(7, 2, 1, f"predict_proba for stroke = 1; class index {rec['stroke1_class_index']}; identity link; probability scale")
cell(7, 3, 1, f"{rec['background_size']} rows from the oversampled training set, shap.sample, seed {rec['background_seed']}")
cell(7, 4, 1, f"nsamples {rec['nsamples']}; l1_reg False; numpy seed {rec['explained_sample_seed']}")
cell(7, 5, 1, f"Wrapper function passes the {split['n_features']} encoded columns to the fitted pipeline")
for r in range(2, 6):
    cell(7, r, 2, "Not used")
cell(7, 1, 2, "Not used")
fill("Chosen route and implementation:",
     f"Chosen route and implementation: Kernel SHAP using shap.KernelExplainer (shap {rec['shap_version']}) on a "
     f"wrapper around the fitted pipeline. Background selection: {rec['background_size']} rows sampled from the "
     f"oversampled training set, seed {rec['background_seed']}. Held-out sample size: {rec['explained_n']} rows, being "
     f"the Task 3 case plus a seeded random sample of other test rows ({rec['explained_positive']} with stroke = 1). "
     f"Computational settings: nsamples {rec['nsamples']}, no L1 regularisation. Deep SHAP was not used.")
warn_text = "the optimiser reached 500 iterations and raised a ConvergenceWarning during fitting, so training had not " \
            "fully converged" if m6["fit_warnings"] else "no warnings were raised during fitting"
fill("Compatibility and checks:",
     f"Compatibility and checks: output shape {tuple(rec['shap_shape'])}; selected class stroke = 1 "
     f"(index {rec['stroke1_class_index']}, classes_ {rec['model_classes']}); reconstruction residual "
     f"{rec['max_abs_residual']:.2e} against tolerance {rec['tolerance']:.0e}, passed; runtime "
     f"{rec['runtime_seconds']:.0f} seconds; {'no warnings' if not rec['warnings'] else 'warnings'} from the explainer"
     f"{'' if not rec['warnings'] else ': ' + '; '.join(rec['warnings'])}. For the model, {warn_text}. This is reported "
     f"as found, and iterations were not raised because no tuning was done. The same warning appeared when the MLP was "
     f"refitted in the threshold cross-validation folds. The same "
     f"preprocessing objects were used for training, prediction and explanation. Kernel SHAP estimates SHAP values by "
     f"sampling, so the values are approximate, although additivity holds by construction.")
fill("[Insert its held-out performance metrics",
     f"Held-out performance on the same {split['n_test']} test rows at the MLP's own threshold of "
     f"{tf(m6['threshold'])}, chosen on training folds by the same rule: accuracy {f3(m6['accuracy'])}, "
     f"balanced accuracy {f3(m6['balanced_accuracy'])}, precision {f3(m6['precision_stroke1'])}, recall "
     f"{f3(m6['recall_stroke1'])}, F1 {f3(m6['f1_stroke1'])} (support {m6['positive_support']}), ROC AUC "
     f"{f3(m6['roc_auc'])}, average precision {f3(m6['average_precision'])}. Confusion matrix: TN {m6['tn']}, FP "
     f"{m6['fp']}, FN {m6['fn']}, TP {m6['tp']}. At 0.5 its recall was {f3(m6h['recall_stroke1'])} and precision "
     f"{f3(m6h['precision_stroke1'])}. The threshold is very low because most MLP scores sit close to 0. Both plots "
     f"below are in units of predicted probability of stroke = 1.")
fig4_par, fig5_par = find("Figure 4:"), find("Figure 5:")
picture("Figure 4:", "task6/figure4_local_force_plot.png", 6.2)
picture("Figure 5:", "task6/figure5_global_summary_plot.png", 5.2)
mlp_local = comp["case_mlp_top5"]
sign_rows = {r["feature"]: r for r in comp["case_rf_top5_sign_agreement"]}
interp = find("Captions and interpretation:")
for prefix_par, text in (
    (fig4_par, f"Figure 4. Kernel SHAP force plot for the MLP, same case as Figure 2 (test_row {comp['case_test_row']}). "
               f"Base value {f3(rec['base_value'])}, output {f3(comp['case_mlp_probability'])}, in predicted probability "
               f"of stroke = 1. The axis runs past 1 only because the up and down pushes are drawn side by side."),
    (fig5_par, f"Figure 5. Kernel SHAP summary plot for the MLP on {rec['explained_n']} held-out rows "
               f"({rec['explained_positive']} with stroke = 1). The x axis is the change in predicted probability of "
               f"stroke = 1. Age again has the widest spread."),
):
    new = copy.deepcopy(interp._p)
    prefix_par._p.addnext(new)
    set_runs(docx.text.paragraph.Paragraph(new, interp._parent), text)
set_runs(interp,
         f"The MLP also predicts stroke = 1 for this case, with a higher probability of "
         f"{f3(comp['case_mlp_probability'])}. Its largest contribution is {mlp_local[0]['feature']} = 1 "
         f"({signed(mlp_local[0]['shap'])}), followed by avg_glucose_level ({signed(mlp_local[1]['shap'])}) and age "
         f"({signed(mlp_local[2]['shap'])}). Agreement with Task 2 is partial. Both models push the case up through "
         f"glucose, but heart_disease is much smaller in the MLP ({signed(sign_rows['heart_disease']['shap_mlp'])} vs "
         f"{signed(sign_rows['heart_disease']['shap_rf'])}), and bmi changes sign "
         f"({signed(sign_rows['bmi']['shap_mlp'])} vs {signed(sign_rows['bmi']['shap_rf'])}). Over all "
         f"{split['n_features']} features, the signs match for {comp['case_all_feature_sign_agreement_n']}. Globally, "
         f"age is first for both models, but the MLP gives more weight to work_type_Private and smoking_status_Unknown. "
         f"Its summary plot also spreads hypertension to the negative side, unlike the RandomForest. That disagreement "
         f"should be read with care, given the MLP's weaker ranking and non-converged training.")

# ---- Section 8 ----
cell(8, 1, 1, "RandomForest; shap.TreeExplainer")
cell(8, 1, 2, "StandardScaler + MLPClassifier; shap.KernelExplainer")
for col, m in ((1, m2), (2, m6)):
    cell(8, 2, col, f"Recall {f3(m['recall_stroke1'])}; precision {f3(m['precision_stroke1'])} "
                    f"(threshold {tf(m['threshold'])})")
    cell(8, 3, col, f"F1 {f3(m['f1_stroke1'])}; balanced accuracy {f3(m['balanced_accuracy'])}; ROC AUC {f3(m['roc_auc'])}")
    cell(8, 4, col, f"Average precision {f3(m['average_precision'])}; prevalence {pct(m['prevalence_test'])}")
cell(8, 5, 1, f"{f2(case['predicted_probability'])}; predicted probability of stroke = 1")
cell(8, 5, 2, f"{f3(comp['case_mlp_probability'])}; predicted probability of stroke = 1")
top3_rf = [r["feature"] for r in glob["top5"][:3]]
top3_mlp = [r["feature"] for r in comp["mlp_top5_global"][:3]]
top3_rf_same = [r["feature"] for r in comp["rf_top5_same_rows"][:3]]
same_note = "same three on the Task 6 rows" if top3_rf == top3_rf_same else f"on the Task 6 rows: {', '.join(top3_rf_same)}"
cell(8, 6, 1, f"{', '.join(top3_rf)} ({same_note})")
cell(8, 6, 2, ", ".join(top3_mlp))
cell(8, 7, 1, f"All {ver['n_explained']} test rows; {ver['shap_seconds']:.0f} s")
cell(8, 7, 2, f"{rec['explained_n']} test rows; {rec['runtime_seconds']:.0f} s")
cell(8, 8, 1, f"Max {ver['max_abs_residual']:.2e}")
cell(8, 8, 2, f"Max {rec['max_abs_residual']:.2e}")
fill("[Explain agreement or disagreement",
     f"At their chosen thresholds the models give the same hard prediction for "
     f"{comp['hard_prediction_agreement_test'] * 100:.1f}% of test rows "
     f"({comp['hard_prediction_agreement_test_at_0_5'] * 100:.1f}% at 0.5), and their scores have a Spearman correlation of "
     f"{comp['spearman_test_scores_rf_vs_mlp']:.2f}. On the same {rec['explained_n']} rows, the Spearman correlation of "
     f"mean absolute SHAP across features is {comp['spearman_mean_abs_shap_rf_vs_mlp_same_rows']:.2f}, so importance "
     f"rankings agree only moderately. The magnitudes can be compared because both are on the probability scale and "
     f"the same encoded features, but the explained samples differ in the table (Task 2 used all test rows), the "
     f"reference differs (tree-path coverage vs a 100-row background), and the base values are "
     f"{f3(ver['base_value'])} and {f3(rec['base_value'])}. Differences may therefore reflect the models, the explainers "
     f"or the background. No model should be favoured because its explanation looks more familiar.")
fill("[Discuss dataset provenance",
     f"Provenance and representativeness are unknown: the supplied files give no population, sites or collection "
     f"period, so results may not carry over to any real clinical group. The dataset is retrospective and there was no "
     f"prospective validation. The thresholds came from out-of-fold training predictions, not a separate validation "
     f"set, and were checked on one test set with only {m2['positive_support']} positives, so the test metrics are "
     f"noisy. Classes are imbalanced ({eda['stroke_pct_positive']}% positive), and oversampling "
     f"repeats minority rows, which can encourage overfitting. Missingness also matters: bmi is blank for "
     f"{int(miss.loc['bmi', 'missing_n'])} rows with a higher stroke share, and Section 5 suggests that the imputed "
     f"value may carry that pattern into the model, "
     f"while smoking is Unknown for {int(miss.loc['smoking_status (Unknown)', 'missing_n'])} rows. Probability "
     f"calibration was not assessed. Scores come from models trained on balanced data, so they should be treated as "
     f"uncalibrated. Subgroup performance was not assessed; with only {m2['positive_support']} positive test rows, the "
     f"subgroups would be very small. The recorded stroke may also precede the measured values, so the data cannot "
     f"support a temporal reading. Correlated features share attribution, as shown by age and ever_married, and "
     f"background selection affects attribution, as shown next.")
sign_note = "all kept their sign" if all(r["sign_same"] for r in rep["case_top5_first_vs_second"]) else "did not all keep their sign"
age_rep = next(r for r in rep["case_top5_first_vs_second"] if r["feature"] == "age")
fill("[If feasible, repeat explanations",
     f"The repeat was quick, so Kernel SHAP was run again with a second background of {rep['background_size']} training "
     f"rows (seed {rep['second_background_seed']}), using the same MLP and the same {rep['explained_n']} rows. The top "
     f"five global features overlapped on {rep['top5_overlap']} of 5, and the Spearman correlation of mean absolute "
     f"SHAP was {rep['spearman_mean_abs_shap']:.2f}. For the Task 3 case, the five largest contributions {sign_note}, "
     f"and {rep['case_sign_changes_all_features']} of {split['n_features']} features changed sign overall. Magnitudes "
     f"did move: the age contribution went from {signed(age_rep['first'])} to {signed(age_rep['second'])}, and the base "
     f"value went from {f3(rep['base_value_first'])} to {f3(rep['base_value_second'])}. The ranking was stable in this "
     f"one repeat, but local values depended on the background. The RandomForest was not repeated because its path-"
     f"dependent TreeExplainer takes no background.")
conclusion = (
    f"On threshold-free ranking the RandomForest performed better than the MLP (ROC AUC {f3(m2['roc_auc'])} vs "
    f"{f3(m6['roc_auc'])}, average precision {f3(m2['average_precision'])} vs {f3(m6['average_precision'])}). With "
    f"thresholds chosen on training folds, recall rose from {f3(m2h['recall_stroke1'])} to {f3(m2['recall_stroke1'])} "
    f"for the RandomForest and from {f3(m6h['recall_stroke1'])} to {f3(m6['recall_stroke1'])} for the MLP, but "
    f"precision stayed low ({f3(m2['precision_stroke1'])} and {f3(m6['precision_stroke1'])}), so most flagged rows "
    f"were false alarms. Local SHAP showed that the selected case "
    f"reached {f2(case['predicted_probability'])} mainly through heart disease and glucose. Global SHAP ranked age, "
    f"glucose and bmi highest. With the MLP, age stayed first, but the same case scored "
    f"{f3(comp['case_mlp_probability'])} and its bmi contribution changed sign. The main constraint is that both "
    f"explanations describe models trained on balanced, retrospective data with uncalibrated probabilities, tested "
    f"once on {m2['positive_support']} positives, so they describe model behaviour and not stroke mechanisms or "
    f"individual risk."
)
assert 100 <= len(conclusion.split()) <= 150, len(conclusion.split())
assert m2["precision_stroke1"] < 0.5 and m6["precision_stroke1"] < 0.5   # "most flagged rows were false alarms"
assert m2["recall_stroke1"] > m2h["recall_stroke1"] and m6["recall_stroke1"] > m6h["recall_stroke1"]   # "recall rose"
fill("[In approximately 100", conclusion)
fill("Individual or group contributions:",
     "Individual or group contributions: " + student("contributions", "enter names and tasks") + ". Assistance and "
     "tools used: " + student("tool_disclosure", "confirm final wording under course policy. Suggested: Claude Code "
     "was used to run the analysis, write the code and draft this report, and all reported results come from code "
     "that was actually run") + ".")

# ---- Section 9 ----
fill("Use the citation style required",
     "References are numbered." if info.get("citation_style_confirmed") else
     "References are numbered. [Student: change the style if the course requires another one.]")
refs = find("[1]")
set_runs(refs, "[1] fedesoriano. Stroke Prediction Dataset. Kaggle. "
               "https://www.kaggle.com/fedesoriano/stroke-prediction-dataset")
ref2 = copy.deepcopy(refs._p)
refs._p.addnext(ref2)
set_runs(docx.text.paragraph.Paragraph(ref2, refs._parent),
         "[2] Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In "
         "Advances in Neural Information Processing Systems 30, pp. 4765-4774.")
fill("Environment:",
     f"Environment: Python {ver['python']}, Windows 11, {versions()}; full list in requirements.txt. Seeds: "
     f"random_state=2021 for the split, oversampling, both models, background and explained sample; 2022 for the second "
     f"background only. Data fingerprint or version: SHA-256 {eda['file_sha256']}. Configuration: RandomForest defaults "
     f"with random_state=2021; StandardScaler + MLPClassifier defaults with max_iter=500; bmi median "
     f"{split['bmi_train_median']}; decision thresholds {tf(t_rf['threshold'])} (RandomForest) and "
     f"{tf(t_mlp['threshold'])} (MLP) from work/02a_select_threshold.py (max F2, 5-fold out-of-fold, seed "
     f"{thr['cv_seed']}); TreeExplainer with tree_path_dependent; KernelExplainer with "
     f"{rec['background_size']} background rows and nsamples {rec['nsamples']}. Split indices are reproduced by "
     f"train_test_split with test_size=0.20, stratified, seed 2021. Encoded feature order: "
     f"{', '.join(split['feature_names'])}.")
fill("Evidence files:",
     "Evidence files: EDA outputs in outputs/eda/; trained pipeline in outputs/task2/rf_model.joblib, "
     "outputs/task2/preprocessing.joblib and outputs/task6/mlp_model.joblib; metrics in outputs/task2/metrics.json and "
     "outputs/task6/metrics.json; threshold selection in work/02a_select_threshold.py, "
     "outputs/threshold/thresholds.json and outputs/threshold/oof_predictions.csv; confusion matrix in outputs/task2/confusion_matrix.png; force plot in "
     "outputs/task3/force_plot_case.png and .html; summary plot in outputs/task4/summary_plot.png; alternative model "
     "plots in outputs/task6/; explanation arrays in outputs/task2/shap_values_test.npz and "
     "outputs/task6/kernel_shap_values.npz; three-slide file in outputs/slides/Stroke_SHAP_Slides.pptx; environment in "
     "requirements.txt; scripts in work/; log in outputs/progress_log.md.")

slides_done = (OUT / "slides/Stroke_SHAP_Slides.pptx").exists()
done = ["☐ Task 2", "☐ Task 3", "☐ Task 4", "☐ Task 6", "☐ No invented"]
if slides_done:
    done.append("☐ Task 5")
if info.get("code_source") and info.get("repository"):
    done.append("☐ Task 1")
texts_now = [p.text for p in d.paragraphs] + [c.text for t in d.tables for row in t.rows for c in row.cells]
if not any("[Student:" in t for t in texts_now):
    done.append("☐ All placeholders")
for par in d.paragraphs:
    if any(par.text.startswith(k) for k in done):
        set_runs(par, par.text.replace("☐", "☑", 1))

# ---- Guard: no template placeholder may survive ----
texts = [p.text for p in d.paragraphs]
texts += [c.text for t in d.tables for row in t.rows for c in row.cells]
left = []
for text in texts:
    for match in re.findall(r"\[[^\]]*\]?", text):
        if not (match.startswith("[Student:") or re.fullmatch(r"\[\d\]", match)
                or re.fullmatch(r"\[[0, 1]+\]", match)):
            left.append(match[:70])
blank = [t for t in texts if "___" in t]
if left or blank:
    raise SystemExit(f"Placeholders left: {left} {blank}")

# The template starts each Heading 1 after an empty paragraph holding a page break. When the page before
# is exactly full, that paragraph spills onto a blank page. Moving the break onto the heading itself
# ("page break before") gives the same layout without blank pages.
for par in list(d.paragraphs):
    if par.text == "" and 'w:br w:type="page"' in par._p.xml:
        nxt = par._p.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            following = docx.text.paragraph.Paragraph(nxt, par._parent)
            if following.style.name == "Heading 1":
                following.paragraph_format.page_break_before = True
                par._p.getparent().remove(par._p)

d.save(TARGET)
print("saved", TARGET.name, "| abstract", len(abstract.split()), "words | conclusion", len(conclusion.split()), "words")
