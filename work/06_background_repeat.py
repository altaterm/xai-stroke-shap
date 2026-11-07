"""Optional check from the template: repeat Kernel SHAP with a second training background.

Same MLP, same explained rows and nsamples. Only the background sample changes.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import shap

from common import OUT, prepare_data

T6 = OUT / "task6"
SECOND_SEED = 2022

d = prepare_data()
names = d["feature_names"]
X_test = d["X_test"].reset_index(drop=True)
model = joblib.load(T6 / "mlp_model.joblib")
stroke_idx = int(np.where(model.classes_ == 1)[0][0])
first = np.load(T6 / "kernel_shap_values.npz", allow_pickle=False)
sv1, rows = first["shap_values"], first["test_rows"]
record = json.loads((T6 / "explainer_record.json").read_text(encoding="utf-8"))
X_exp = X_test.iloc[rows].to_numpy()


def predict_stroke(a):
    return model.predict_proba(pd.DataFrame(a, columns=names))[:, stroke_idx]


background = shap.sample(d["X_train_res"], record["background_size"], random_state=SECOND_SEED)
explainer = shap.KernelExplainer(predict_stroke, background.to_numpy(), link="identity")
np.random.seed(SECOND_SEED)
t0 = time.perf_counter()
sv2 = np.asarray(explainer.shap_values(X_exp, nsamples=record["nsamples"], l1_reg=False, silent=True))
runtime = time.perf_counter() - t0
base2 = float(np.ravel(explainer.expected_value)[0])
resid = np.abs(base2 + sv2.sum(axis=1) - predict_stroke(X_exp))

m1 = pd.Series(np.abs(sv1).mean(axis=0), index=names).sort_values(ascending=False)
m2 = pd.Series(np.abs(sv2).mean(axis=0), index=names).sort_values(ascending=False)
case1 = pd.Series(sv1[0], index=names)
case2 = pd.Series(sv2[0], index=names)
top5_case = case1.abs().sort_values(ascending=False).index[:5]
result = {
    "second_background_seed": SECOND_SEED, "background_size": record["background_size"],
    "explained_n": len(rows), "nsamples": record["nsamples"], "runtime_seconds": runtime,
    "base_value_first": record["base_value"], "base_value_second": base2,
    "max_abs_residual_second": float(resid.max()),
    "top5_first": list(m1.index[:5]), "top5_second": list(m2.index[:5]),
    "top5_overlap": len(set(m1.index[:5]) & set(m2.index[:5])),
    "top3_first": list(m1.index[:3]), "top3_second": list(m2.index[:3]),
    "spearman_mean_abs_shap": float(m1.corr(m2.reindex(m1.index), method="spearman")),
    "case_top5_first_vs_second": [
        {"feature": f, "first": float(case1[f]), "second": float(case2[f]),
         "sign_same": bool(np.sign(case1[f]) == np.sign(case2[f]))} for f in top5_case],
    "case_sign_changes_all_features": int((np.sign(case1) != np.sign(case2)).sum()),
    "note": "RandomForest TreeExplainer uses tree_path_dependent and takes no background, so only Task 6 was repeated.",
}
(T6 / "background_repeat.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=1))
