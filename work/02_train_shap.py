"""Task 2: RandomForest, test metrics, TreeExplainer and additivity check."""
import json
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier

from common import OUT, SEED, binary_metrics, prepare_data

T2 = OUT / "task2"
T2.mkdir(parents=True, exist_ok=True)
# Threshold chosen by 02a_select_threshold.py on out-of-fold training predictions only
THRESHOLD = json.loads((OUT / "threshold" / "thresholds.json").read_text(encoding="utf-8"))["models"]["random_forest"]["threshold"]
TOL = 1e-6

d = prepare_data()
X_res, y_res, X_test, y_test = d["X_train_res"], d["y_train_res"], d["X_test"], d["y_test"]
names = d["feature_names"]

# Leakage guards
assert "id" not in names and "stroke" not in names
assert not set(d["X_train"].index) & set(X_test.index)
assert X_test["bmi"].notna().all() and d["X_train"]["bmi"].notna().all()
assert len(y_res) > len(d["y_train"]) and len(y_test) == len(X_test)

split = {
    "n_total": len(d["raw"]),
    "n_train": len(d["y_train"]), "train_pos": int(d["y_train"].sum()),
    "n_test": len(y_test), "test_pos": int(y_test.sum()),
    "n_train_oversampled": len(y_res), "train_oversampled_pos": int(y_res.sum()),
    "bmi_train_median": d["bmi_median"],
    "bmi_missing_train": int(d["X_train_raw"]["bmi"].isna().sum()),
    "bmi_missing_test": int(d["X_test_raw"]["bmi"].isna().sum()),
    "gender_other_in_train": int((d["X_train_raw"]["gender"] == "Other").sum()),
    "gender_other_in_test": int((d["X_test_raw"]["gender"] == "Other").sum()),
    "n_features": len(names), "feature_names": names, "seed": SEED, "validation_split": "none",
}

model = RandomForestClassifier(random_state=SEED)
t0 = time.perf_counter()
model.fit(X_res, y_res)
fit_s = time.perf_counter() - t0

proba = model.predict_proba(X_test)
stroke_idx = int(np.where(model.classes_ == 1)[0][0])
p1 = proba[:, stroke_idx]
y_pred = (p1 >= THRESHOLD).astype(int)
metrics = binary_metrics(y_test, y_pred, p1, THRESHOLD)
metrics["model"] = "RandomForestClassifier(random_state=2021), other parameters default"
metrics["fit_seconds"] = fit_s
metrics["threshold_rule"] = "max F2 on 5-fold out-of-fold training predictions (outputs/threshold/thresholds.json)"
metrics["metrics_at_0_5"] = binary_metrics(y_test, (p1 >= 0.5).astype(int), p1, 0.5)

explainer = shap.TreeExplainer(model)
t0 = time.perf_counter()
sv_raw = explainer.shap_values(X_test)
shap_s = time.perf_counter() - t0
sv_raw = np.asarray(sv_raw)
assert sv_raw.shape == (len(X_test), len(names), 2), sv_raw.shape   # (rows, features, classes)
sv = sv_raw[:, :, stroke_idx]
base = float(np.asarray(explainer.expected_value)[stroke_idx])

recon = base + sv.sum(axis=1)
resid = np.abs(recon - p1)
verification = {
    "model_classes": [int(c) for c in model.classes_],
    "stroke1_class_index": stroke_idx,
    "shap_version": shap.__version__, "sklearn_version": sklearn.__version__,
    "python": sys.version.split()[0],
    "shap_values_raw_shape": list(sv_raw.shape),
    "shap_output_format": "single ndarray (rows, features, classes), not a list per class",
    "explainer_expected_value": [float(v) for v in np.asarray(explainer.expected_value)],
    "explainer_model_output": str(explainer.model_output),
    "explainer_feature_perturbation": str(explainer.feature_perturbation),
    "explainer_data_passed": explainer.data is not None,
    "output_explained": "predicted probability of stroke = 1 (no link function)",
    "base_value": base,
    "n_explained": int(sv.shape[0]),
    "feature_names_match_model": list(model.feature_names_in_) == list(X_test.columns) == names,
    "n_features_model": int(model.n_features_in_),
    "max_abs_residual": float(resid.max()), "mean_abs_residual": float(resid.mean()),
    "tolerance": TOL, "reconstruction_pass": bool(resid.max() <= TOL),
    "example_row0": {"base_plus_sum": float(recon[0]), "model_output": float(p1[0])},
    "shap_seconds": shap_s,
    "train_mean_oversampled_p1": float(model.predict_proba(X_res)[:, stroke_idx].mean()),
}

joblib.dump(model, T2 / "rf_model.joblib", compress=3)
joblib.dump({"encoder": d["encoder"], "bmi_median": d["bmi_median"], "feature_names": names},
            T2 / "preprocessing.joblib")
X_test.assign(y_true=y_test.values, p_stroke=p1, y_pred=y_pred).reset_index(drop=True) \
    .to_csv(T2 / "test_set_encoded_with_predictions.csv", index_label="test_row")
np.savez_compressed(T2 / "shap_values_test.npz", shap_values=sv, base_value=base,
                    feature_names=np.array(names))
(T2 / "split_info.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
(T2 / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
pd.Series({k: v for k, v in metrics.items() if not isinstance(v, dict)}).to_csv(T2 / "metrics.csv", header=["value"])
(T2 / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
            color="white" if v > cm.max() / 2 else "#1F2933")
ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
ax.set_title(f"RandomForest, test set, threshold {THRESHOLD:.2f}", fontsize=11)
fig.tight_layout()
fig.savefig(T2 / "confusion_matrix.png", dpi=170, facecolor="white")

for k in ("n_train", "train_pos", "n_test", "test_pos", "n_train_oversampled", "bmi_train_median",
          "bmi_missing_train", "bmi_missing_test", "gender_other_in_train", "n_features"):
    print(k, split[k])
print(json.dumps(metrics, indent=1))
print(json.dumps(verification, indent=1))
print(pd.Series(p1).describe())
print("predicted positives:", y_pred.sum(), " p>=.3:", (p1 >= .3).sum())
print(names)
