"""Phase 8: independent checks on the finished deliverables."""
import hashlib
import json
import re
import subprocess
import sys

import docx
import numpy as np
import pandas as pd
from pptx import Presentation

from common import OUT, ROOT, prepare_data

d = prepare_data()
results = {}

# Leakage
tr, te, res = d["X_train"], d["X_test"], d["X_train_res"]
results["train_test_rows_disjoint"] = not set(tr.index) & set(te.index)
results["split_covers_all_rows"] = len(tr) + len(te) == len(d["raw"])
results["bmi_median_is_train_median"] = float(d["X_train_raw"].bmi.median()) == d["bmi_median"]
results["median_differs_from_all_data_or_equal_noted"] = float(d["raw"].bmi.median())
results["test_row_count_unchanged"] = len(te) == 1022
key = list(tr.columns)
results["no_test_row_in_oversampled_train"] = len(res.round(6).merge(te.round(6), on=key, how="inner")) == 0
results["oversampled_rows_all_from_train"] = len(res.round(6).drop_duplicates()) == len(tr.round(6).drop_duplicates())
results["id_stroke_not_predictors"] = not {"id", "stroke"} & set(key)
results["test_class_balance_kept"] = float(d["y_test"].mean())

# Report
rep = docx.Document(ROOT / "Stroke_SHAP_Report_Completed.docx")
tmp = docx.Document(ROOT / "Stroke_SHAP_Report_Template.docx")
text = [p.text for p in rep.paragraphs] + [c.text for t in rep.tables for r in t.rows for c in r.cells]
full = "\n".join(text)
def heads(document):
    return [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
results["headings_kept_except_how_to_use"] = heads(rep) == [h for h in heads(tmp) if h != "How to use this template"]
results["tables"] = (len(rep.tables), len(tmp.tables))
results["inline_images"] = len(rep.inline_shapes)
results["figure_captions"] = sorted(set(re.findall(r"^Figure \d\.", full, flags=re.M)))
results["student_markers_left"] = len(set(re.findall(r"\[Student:[^\]]*\]", full)))
results["other_brackets_left"] = sorted(set(re.findall(r"\[[^\]]*\]", full)) - set(re.findall(r"\[Student:[^\]]*\]", full)))
results["checklist"] = [p.text[:32] for p in rep.paragraphs if p.text[:1] in "☐☑" and p.text]

# Slides
prs = Presentation(OUT / "slides/Stroke_SHAP_Slides.pptx")
slides = ["\n".join(s.text_frame.text for s in sl.shapes if s.has_text_frame) for sl in prs.slides]
results["slide_count"] = len(slides)
results["bullets_per_slide"] = [len(sl.placeholders[1].text_frame.paragraphs) for sl in prs.slides]
m, case = json.loads((OUT / "task2/metrics.json").read_text()), json.loads((OUT / "task3/selected_case.json").read_text())
expected = [f"{m['roc_auc']:.3f}", f"{m['recall_stroke1']:.3f}", f"{m['precision_stroke1']:.3f}"]
results["slide1_numbers_match"] = all(v in slides[0] and v in full for v in expected)
results["slide2_case_matches"] = f"{case['predicted_probability']:.2f}" in slides[1] and f"{m['fn']} of {m['positive_support']}" in slides[1]
results["slide3_has_no_digits_or_codes"] = re.findall(r"\d|_", slides[2])
results["slide3_says_percent"] = "%" in slides[2]
banned = r"—|leverage|robust|delve|seamless|key takeaway|at a glance"
results["slide_banned_words"] = re.findall(banned, "\n".join(slides), flags=re.I)
results["report_em_dash_outside_template_cell"] = full.count("—")

# Threshold step
thr_path = OUT / "threshold/thresholds.json"
results["thresholds_file_exists"] = thr_path.exists()
thr = json.loads(thr_path.read_text())
oof = pd.read_csv(OUT / "threshold/oof_predictions.csv")
results["oof_rows_are_training_rows_only"] = (len(oof) == len(d["y_train"]) and
                                              (oof.y_true.to_numpy() == d["y_train"].to_numpy()).all())
results["folds_heldout_cover_train_once"] = sum(f["n_heldout"] for f in thr["folds"]) == len(d["y_train"])
results["fold_heldout_not_oversampled"] = all(f["n_train_oversampled"] > f["n_train"] for f in thr["folds"])
results["flagged_share_under_half"] = all(v["oof_share_flagged"] <= 0.5 for v in thr["models"].values())
m6 = json.loads((OUT / "task6/metrics.json").read_text())
results["test_thresholds_match_file"] = (m["threshold"] == thr["models"]["random_forest"]["threshold"] and
                                         m6["threshold"] == thr["models"]["mlp"]["threshold"])
results["case_row_still_413"] = case["test_row"] == 413 and case["case_type"] == "TP"


def blob_hash(path, ref="before-threshold-update"):
    old = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(old).hexdigest() == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


results["shap_arrays_unchanged_since_tag"] = all(blob_hash(p) for p in
                                                 ("outputs/task2/shap_values_test.npz",
                                                  "outputs/task6/kernel_shap_values.npz"))
results["plots_unchanged_since_tag"] = all(blob_hash(p) for p in
                                           ("outputs/task3/force_plot_case.png", "outputs/task4/summary_plot.png",
                                            "outputs/task6/figure4_local_force_plot.png",
                                            "outputs/task6/figure5_global_summary_plot.png"))
results["slides_no_0_5"] = re.findall(r"\b0\.5\b(?!\d)", "\n".join(slides))
results["report_0_5_mentions"] = [t[:90] for t in text if re.search(r"\b0\.5\b(?!\d)", t)]
highlighted = [r.text for p in rep.paragraphs for r in p.runs if r.font.highlight_color is not None]
results["highlighted_runs_are_markers_only"] = all(t.startswith("[Student:") for t in highlighted)

wording = r"caus\w*|future|risk\w*|diagnos\w*|treat\w*|predict\w* stroke|protective|medication|prevent\w*"
hits = {}
for name, body in (("report", full), ("slides", "\n".join(slides))):
    hits[name] = sorted({s.strip() for s in re.findall(r"[^.\n]*\b(?:" + wording + r")\b[^.\n]*", body, flags=re.I)})
for k, v in results.items():
    print(k, ":", v)
for name, lines in hits.items():
    print("==", name)
    for line in lines:
        print(" -", line[:170])
(OUT / "final_checks.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
ok = all(results[k] is True for k in ("train_test_rows_disjoint", "split_covers_all_rows", "bmi_median_is_train_median",
         "no_test_row_in_oversampled_train", "id_stroke_not_predictors", "headings_kept_except_how_to_use",
         "thresholds_file_exists", "oof_rows_are_training_rows_only", "folds_heldout_cover_train_once",
         "fold_heldout_not_oversampled", "flagged_share_under_half", "test_thresholds_match_file",
         "case_row_still_413", "shap_arrays_unchanged_since_tag", "plots_unchanged_since_tag",
         "highlighted_runs_are_markers_only")) \
    and results["slide_count"] == 3 and not results["slides_no_0_5"]
sys.exit(0 if ok else 1)
