"""Task 3: force plot for one held-out case, using the saved Task 2 outputs.

Selection rule (fixed in code): the true-positive test case with the highest
predicted probability, ties broken by the lowest test_row.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import OUT, prepare_data

T2, T3 = OUT / "task2", OUT / "task3"
T3.mkdir(parents=True, exist_ok=True)

d = prepare_data()
names = d["feature_names"]
pred = pd.read_csv(T2 / "test_set_encoded_with_predictions.csv", index_col="test_row")
npz = np.load(T2 / "shap_values_test.npz", allow_pickle=False)
sv, base = npz["shap_values"], float(npz["base_value"])
threshold = json.loads((T2 / "metrics.json").read_text())["threshold"]

tp = pred[(pred.y_true == 1) & (pred.y_pred == 1)]
row = int(tp.sort_values("p_stroke", ascending=False, kind="stable").index[0])
p = float(pred.loc[row, "p_stroke"])
actual, predicted = int(pred.loc[row, "y_true"]), int(pred.loc[row, "y_pred"])
case_type = {(1, 1): "TP", (0, 1): "FP", (0, 0): "TN", (1, 0): "FN"}[(actual, predicted)]
raw = d["X_test_raw"].iloc[row]

contrib = pd.DataFrame({"feature": names, "encoded_value": pred.loc[row, names].to_numpy(float),
                        "shap": sv[row]})
contrib["abs_shap"] = contrib["shap"].abs()
contrib = contrib.sort_values("abs_shap", ascending=False).reset_index(drop=True)
contrib.to_csv(T3 / "case_contributions_all_features.csv", index=False)

top3 = contrib.head(3)
rest = float(contrib["shap"].iloc[3:].sum())
ups = contrib[contrib["shap"] > 0].head(3)
downs = contrib[contrib["shap"] < 0].head(3)
case = {
    "selection_rule": "true-positive test case with the highest predicted probability; ties by lowest test_row",
    "n_true_positives_available": len(tp),
    "tp_candidates": {int(i): float(v) for i, v in tp["p_stroke"].items()},
    "test_row": row, "actual_class": actual, "predicted_class": predicted, "threshold": threshold,
    "predicted_probability": p, "base_value": base, "case_type": case_type,
    "sum_shap": float(sv[row].sum()), "base_plus_sum": base + float(sv[row].sum()),
    "residual": abs(base + float(sv[row].sum()) - p),
    "raw_values": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                   for k, v in raw.items()},
    "bmi_was_imputed": bool(pd.isna(raw["bmi"])),
    "top3": top3[["feature", "encoded_value", "shap"]].to_dict("records"),
    "remaining_features_sum": rest,
    "largest_up": ups[["feature", "encoded_value", "shap"]].to_dict("records"),
    "largest_down": downs[["feature", "encoded_value", "shap"]].to_dict("records"),
    "output_units": "predicted probability of stroke = 1",
}
(T3 / "selected_case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")

shown = pd.Series(np.round(pred.loc[row, names].to_numpy(float), 2), index=names)
# Only the largest contributions are labelled (shap threshold 0.15, per side) so text stays readable.
# Every feature is still drawn, and all values are in the CSV and the HTML.
fig = shap.force_plot(base, sv[row], shown, matplotlib=True, show=False, figsize=(16, 3.4),
                      text_rotation=0, out_names="P(stroke = 1)", contribution_threshold=0.15)
fig.text(0.5, 0.2, f"RandomForest, test_row {row}. Base value = {base:.3f}, model output = {p:.2f}. "
         "Units: predicted probability of stroke = 1.\n"
         "Red pushes the output up, blue pushes it down. "
         "Unlabelled segments are small contributions.", ha="center", va="top", fontsize=12)
fig.savefig(T3 / "force_plot_case.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
html = shap.force_plot(base, sv[row], shown, out_names="P(stroke = 1)")
shap.save_html(str(T3 / "force_plot_case.html"), html)
print(json.dumps(case, indent=1, default=str))
print(contrib.head(10).to_string())
