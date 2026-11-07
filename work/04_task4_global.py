"""Task 4: summary (beeswarm) plot and top-5 table on the whole held-out test set."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import OUT, prepare_data

T2, T3, T4 = OUT / "task2", OUT / "task3", OUT / "task4"
T4.mkdir(parents=True, exist_ok=True)
MAX_DISPLAY = 15

d = prepare_data()
X_test, names = d["X_test"].reset_index(drop=True), d["feature_names"]
y_test = d["y_test"].reset_index(drop=True)
npz = np.load(T2 / "shap_values_test.npz", allow_pickle=False)
sv = npz["shap_values"]
assert sv.shape == X_test.shape


def pattern(col):
    """Mean SHAP for low vs high feature values, measured rather than read off the plot."""
    x, s = X_test[col], sv[:, names.index(col)]
    if set(x.unique()) <= {0.0, 1.0}:
        low, high = x == 0, x == 1
        rule = "value 0 vs value 1"
    else:
        low, high = x <= x.quantile(0.25), x >= x.quantile(0.75)
        rule = f"lowest quarter (<= {x.quantile(0.25):.1f}) vs highest quarter (>= {x.quantile(0.75):.1f})"
    return {"rule": rule, "n_low": int(low.sum()), "n_high": int(high.sum()),
            "mean_shap_low": float(s[low].mean()), "mean_shap_high": float(s[high].mean()),
            "spearman_value_vs_shap": float(pd.Series(x).corr(pd.Series(s), method="spearman")),
            "share_positive_shap": float((s > 0).mean())}


table = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0),
                      "mean_shap": sv.mean(axis=0), "min_shap": sv.min(axis=0),
                      "max_shap": sv.max(axis=0)}) \
    .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
table.insert(0, "rank", table.index + 1)
table.to_csv(T4 / "mean_abs_shap_all_features.csv", index=False)

top5 = [dict(table.iloc[i][["rank", "feature", "mean_abs_shap"]], **pattern(table.feature[i]))
        for i in range(5)]
top5 = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in top5]

case = json.loads((T3 / "selected_case.json").read_text(encoding="utf-8"))
contrib = pd.read_csv(T3 / "case_contributions_all_features.csv")
contrib["case_rank"] = contrib.index + 1
global_rank = dict(zip(table.feature, table["rank"]))
contrib["global_rank"] = contrib.feature.map(global_rank)

corr_pairs = {
    "age_vs_ever_married_Yes": float(X_test.age.corr(X_test.ever_married_Yes)),
    "age_vs_work_type_children": float(X_test.age.corr(X_test.work_type_children)),
    "age_vs_avg_glucose_level": float(X_test.age.corr(X_test.avg_glucose_level)),
    "age_vs_bmi": float(X_test.age.corr(X_test.bmi)),
    "age_vs_hypertension": float(X_test.age.corr(X_test.hypertension)),
    "age_vs_heart_disease": float(X_test.age.corr(X_test.heart_disease)),
}
# Does the model treat the imputed bmi value as a signal that bmi was blank?
imputed = d["X_test_raw"]["bmi"].isna().to_numpy()
bmi_sv = sv[:, names.index("bmi")]
positive = bmi_sv > 0
bmi_imputation = {
    "n_imputed_test": int(imputed.sum()),
    "imputed_value": d["bmi_median"],
    "mean_bmi_shap_imputed": float(bmi_sv[imputed].mean()),
    "mean_bmi_shap_not_imputed": float(bmi_sv[~imputed].mean()),
    "n_positive_bmi_shap": int(positive.sum()),
    "n_positive_bmi_shap_imputed": int((positive & imputed).sum()),
    "bmi_range_of_positive_shap": [float(X_test.bmi[positive].min()), float(X_test.bmi[positive].max())],
    "stroke_share_imputed_test": float(y_test[imputed].mean()),
    "stroke_share_not_imputed_test": float(y_test[~imputed].mean()),
}
glucose_sv = sv[:, names.index("avg_glucose_level")]
glucose_positive = {"n": int((glucose_sv > 0).sum()),
                    "min_glucose": float(X_test.avg_glucose_level[glucose_sv > 0].min())}
bmi_quintiles = pd.Series(bmi_sv).groupby(pd.qcut(X_test.bmi, 5), observed=True).mean()

summary = {
    "bmi_imputation_check": bmi_imputation,
    "glucose_positive_shap": glucose_positive,
    "bmi_mean_shap_by_quintile": {str(k): float(v) for k, v in bmi_quintiles.items()},
    "model": "RandomForestClassifier, Task 2",
    "output": "predicted probability of stroke = 1",
    "explained_sample": "whole held-out test set, no sampling",
    "n_explained": len(X_test), "n_positive": int(y_test.sum()),
    "n_features_total": len(names), "n_features_displayed": MAX_DISPLAY,
    "feature_representation": "encoded columns as they are, no grouping",
    "top5": top5,
    "case_feature_ranks": contrib[["feature", "shap", "case_rank", "global_rank"]].head(8).to_dict("records"),
    "case_age_shap": float(contrib.loc[contrib.feature == "age", "shap"].iloc[0]),
    "case_age_case_rank": int(contrib.loc[contrib.feature == "age", "case_rank"].iloc[0]),
    "pearson_corr_test_set": corr_pairs,
}
(T4 / "global_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

plt.figure()
shap.summary_plot(sv, X_test, max_display=MAX_DISPLAY, show=False, plot_size=(10, 7.5))
ax = plt.gca()
ax.set_xlabel("SHAP value (change in predicted probability of stroke = 1)", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.set_title(f"RandomForest, held-out test set (n = {len(X_test)}, stroke = 1: {int(y_test.sum())}), "
             f"top {MAX_DISPLAY} of {len(names)} encoded features", fontsize=11, loc="left")
plt.savefig(T4 / "summary_plot.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close("all")

print(table.head(12).to_string())
print(json.dumps(bmi_imputation, indent=1), glucose_positive, bmi_quintiles)
