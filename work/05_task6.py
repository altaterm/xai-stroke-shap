"""Task 6: MLP (a different model type) explained with Kernel SHAP.

Same split, preprocessing, predictors and training oversampling as Task 2.
The StandardScaler sits inside the model pipeline because a neural network needs
scaled inputs. It is fitted on the training data only.
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, SEED, binary_metrics, prepare_data

T2, T3, T4, T6 = OUT / "task2", OUT / "task3", OUT / "task4", OUT / "task6"
T6.mkdir(parents=True, exist_ok=True)
# Thresholds chosen by 02a_select_threshold.py on out-of-fold training predictions only
_thresholds = json.loads((OUT / "threshold" / "thresholds.json").read_text(encoding="utf-8"))["models"]
THRESHOLD = _thresholds["mlp"]["threshold"]
RF_THRESHOLD = _thresholds["random_forest"]["threshold"]
BACKGROUND_N = 100
EXPLAIN_N = 200
NSAMPLES = 2048
TOL = 1e-6

d = prepare_data()
X_res, y_res = d["X_train_res"], d["y_train_res"]
X_test, y_test = d["X_test"].reset_index(drop=True), d["y_test"].reset_index(drop=True)
names = d["feature_names"]

fit_warnings = []
model = make_pipeline(StandardScaler(), MLPClassifier(max_iter=500, random_state=SEED))
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.fit(X_res, y_res)
fit_warnings = sorted({f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught})
fit_s = time.perf_counter() - t0
mlp = model[-1]

stroke_idx = int(np.where(model.classes_ == 1)[0][0])
p1 = model.predict_proba(X_test)[:, stroke_idx]
y_pred = (p1 >= THRESHOLD).astype(int)
metrics = binary_metrics(y_test, y_pred, p1, THRESHOLD)
metrics.update(model="StandardScaler + MLPClassifier(max_iter=500, random_state=2021), other parameters default",
               hidden_layer_sizes=list(mlp.hidden_layer_sizes), n_iter=int(mlp.n_iter_),
               fit_seconds=fit_s, fit_warnings=fit_warnings,
               threshold_rule="max F2 on 5-fold out-of-fold training predictions (outputs/threshold/thresholds.json)",
               metrics_at_0_5=binary_metrics(y_test, (p1 >= 0.5).astype(int), p1, 0.5))


def predict_stroke(a):
    """Probability of stroke = 1 for encoded input, as a plain 1-D array."""
    return model.predict_proba(pd.DataFrame(a, columns=names))[:, stroke_idx]


case = json.loads((T3 / "selected_case.json").read_text(encoding="utf-8"))
case_row = int(case["test_row"])

background = shap.sample(X_res, BACKGROUND_N, random_state=SEED)
rng = np.random.RandomState(SEED)
others = np.setdiff1d(np.arange(len(X_test)), [case_row])
rows = np.concatenate([[case_row], np.sort(rng.choice(others, EXPLAIN_N - 1, replace=False))])
X_exp = X_test.iloc[rows]

explainer = shap.KernelExplainer(predict_stroke, background.to_numpy(), link="identity")
np.random.seed(SEED)
run_warnings = []
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sv = np.asarray(explainer.shap_values(X_exp.to_numpy(), nsamples=NSAMPLES, l1_reg=False, silent=True))
run_warnings = sorted({f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught})
shap_s = time.perf_counter() - t0
base = float(np.ravel(explainer.expected_value)[0])

assert sv.shape == (EXPLAIN_N, len(names)), sv.shape
out = predict_stroke(X_exp.to_numpy())
resid = np.abs(base + sv.sum(axis=1) - out)

record = {
    "explainer": "shap.KernelExplainer", "shap_version": shap.__version__,
    "model_classes": [int(c) for c in model.classes_], "stroke1_class_index": stroke_idx,
    "output_explained": "predict_proba for stroke = 1, identity link, probability scale",
    "background_source": "shap.sample of the oversampled training set",
    "background_size": BACKGROUND_N, "background_seed": SEED,
    "background_positive_share": float(y_res.loc[background.index].mean()),
    "explained_sample": "case test_row plus a seeded random sample of other held-out rows",
    "explained_n": EXPLAIN_N, "explained_positive": int(y_test.iloc[rows].sum()),
    "explained_sample_seed": SEED, "nsamples": NSAMPLES, "l1_reg": False,
    "base_value": base, "shap_shape": list(sv.shape), "runtime_seconds": shap_s,
    "warnings": run_warnings, "max_abs_residual": float(resid.max()),
    "mean_abs_residual": float(resid.mean()), "tolerance": TOL,
    "reconstruction_pass": bool(resid.max() <= TOL),
    "mean_p_stroke_on_test": float(p1.mean()),
}

# Local result for the Task 3 case (first explained row)
local = pd.DataFrame({"feature": names, "encoded_value": X_exp.iloc[0].to_numpy(), "shap": sv[0]})
local["abs_shap"] = local["shap"].abs()
local = local.sort_values("abs_shap", ascending=False).reset_index(drop=True)
local.to_csv(T6 / "case_contributions_all_features.csv", index=False)
case_p = float(out[0])

# Compare with Task 2 on the same explained rows, on the same probability scale
rf_npz = np.load(T2 / "shap_values_test.npz", allow_pickle=False)
rf_sv_all = rf_npz["shap_values"]
rf_sv = rf_sv_all[rows]
rf_case = pd.read_csv(T3 / "case_contributions_all_features.csv")
glob = pd.DataFrame({"feature": names, "mlp_mean_abs_shap": np.abs(sv).mean(axis=0),
                     "rf_mean_abs_shap_same_rows": np.abs(rf_sv).mean(axis=0)})
glob["mlp_rank"] = glob.mlp_mean_abs_shap.rank(ascending=False).astype(int)
glob["rf_rank_same_rows"] = glob.rf_mean_abs_shap_same_rows.rank(ascending=False).astype(int)
glob = glob.sort_values("mlp_rank").reset_index(drop=True)
glob.to_csv(T6 / "mean_abs_shap_all_features.csv", index=False)

merged = local.merge(rf_case[["feature", "shap"]], on="feature", suffixes=("_mlp", "_rf"))
sign_agree_top5_rf = merged.loc[merged.feature.isin(rf_case.feature.head(5))]
rf_rows_pred = pd.read_csv(T2 / "test_set_encoded_with_predictions.csv", index_col="test_row")
rf_p = rf_rows_pred["p_stroke"].to_numpy()
comparison = {
    "case_test_row": case_row, "case_mlp_probability": case_p,
    "case_mlp_predicted_class": int(case_p >= THRESHOLD),
    "case_mlp_top5": local[["feature", "encoded_value", "shap"]].head(5).to_dict("records"),
    "case_mlp_age_shap": float(local.loc[local.feature == "age", "shap"].iloc[0]),
    "case_rf_top5_sign_agreement": sign_agree_top5_rf[["feature", "shap_rf", "shap_mlp"]].to_dict("records"),
    "case_all_feature_sign_agreement_n": int((np.sign(merged.shap_mlp) == np.sign(merged.shap_rf)).sum()),
    "mlp_top5_global": glob[["feature", "mlp_mean_abs_shap"]].head(5).to_dict("records"),
    "rf_top5_same_rows": glob.sort_values("rf_rank_same_rows")[["feature", "rf_mean_abs_shap_same_rows"]].head(5).to_dict("records"),
    "spearman_mean_abs_shap_rf_vs_mlp_same_rows":
        float(glob.mlp_mean_abs_shap.corr(glob.rf_mean_abs_shap_same_rows, method="spearman")),
    "spearman_test_scores_rf_vs_mlp": float(pd.Series(rf_p).corr(pd.Series(p1), method="spearman")),
    "hard_prediction_agreement_test": float(np.mean((rf_p >= RF_THRESHOLD) == (p1 >= THRESHOLD))),
    "hard_prediction_agreement_test_at_0_5": float(np.mean((rf_p >= 0.5) == (p1 >= 0.5))),
    "rf_shap_runtime_seconds_all_test":
        json.loads((T2 / "verification.json").read_text())["shap_seconds"],
}

joblib.dump(model, T6 / "mlp_model.joblib", compress=3)
np.savez_compressed(T6 / "kernel_shap_values.npz", shap_values=sv, base_value=base, test_rows=rows,
                    feature_names=np.array(names))
(T6 / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
pd.Series({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}).to_csv(T6 / "metrics.csv", header=["value"])
(T6 / "explainer_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
(T6 / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

shown = pd.Series(np.round(X_exp.iloc[0].to_numpy(float), 2), index=names)
fig = shap.force_plot(base, sv[0], shown, matplotlib=True, show=False, figsize=(16, 3.4),
                      text_rotation=0, out_names="P(stroke = 1)", contribution_threshold=0.15)
fig.text(0.5, 0.2, f"MLP with Kernel SHAP, test_row {case_row}. Base value = {base:.3f}, model output = {case_p:.3f}. "
         "Units: predicted probability of stroke = 1.\n"
         "Red pushes the output up, blue pushes it down. Unlabelled segments are small contributions.",
         ha="center", va="top", fontsize=12)
fig.savefig(T6 / "figure4_local_force_plot.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)

plt.figure()
shap.summary_plot(sv, X_exp, max_display=15, show=False, plot_size=(10, 7.5))
ax = plt.gca()
ax.set_xlabel("Kernel SHAP value (change in predicted probability of stroke = 1)", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.set_title(f"MLP, held-out sample (n = {EXPLAIN_N}, stroke = 1: {record['explained_positive']}), "
             f"top 15 of {len(names)} encoded features", fontsize=11, loc="left")
plt.savefig(T6 / "figure5_global_summary_plot.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close("all")

print(json.dumps(metrics, indent=1))
print(json.dumps(record, indent=1))
print(json.dumps(comparison, indent=1))
print(glob.head(8).to_string())
