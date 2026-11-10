"""Choose a decision threshold per model from out-of-fold predictions on the training split only.

Rule (fixed before looking at any test result): maximise F2 for stroke = 1 on the
5-fold out-of-fold predictions; candidates are the thresholds from
precision_recall_curve; ties go to the higher threshold. Preprocessing and
oversampling are fitted inside each training fold; held-out fold rows are never
oversampled. The test set is not used here.
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, SEED, fit_preprocessing, oversample, prepare_data

TH = OUT / "threshold"
TH.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
MAX_FLAGGED_SHARE = 0.50

MODELS = {
    "random_forest": lambda: RandomForestClassifier(random_state=SEED),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(max_iter=500, random_state=SEED)),
}

d = prepare_data()
X_raw = d["X_train_raw"].reset_index(drop=True)
y = d["y_train"].reset_index(drop=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = {name: np.full(len(y), np.nan) for name in MODELS}
fold_info, fit_warnings = [], {name: set() for name in MODELS}
t0 = time.perf_counter()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
    transform, bmi_median, _, names = fit_preprocessing(X_raw.iloc[tr_idx])
    X_tr, X_va = transform(X_raw.iloc[tr_idx]), transform(X_raw.iloc[va_idx])
    X_tr_res, y_tr_res = oversample(X_tr, y.iloc[tr_idx])          # training fold only
    assert len(X_va) == len(va_idx)                                 # held-out fold untouched
    fold_info.append({"fold": fold, "n_train": len(tr_idx), "n_heldout": len(va_idx),
                      "heldout_pos": int(y.iloc[va_idx].sum()), "n_train_oversampled": len(y_tr_res),
                      "bmi_median_fold": bmi_median})
    for name, make in MODELS.items():
        model = make()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_tr_res, y_tr_res)
        fit_warnings[name] |= {f"{w.category.__name__}: {str(w.message)[:120]}" for w in caught}
        idx = int(np.where(model.classes_ == 1)[0][0])
        oof[name][va_idx] = model.predict_proba(X_va)[:, idx]
    print(f"fold {fold} done", flush=True)
assert not any(np.isnan(v).any() for v in oof.values())

result = {"rule": "maximise F2 (beta=2) for stroke = 1 on out-of-fold training predictions; candidates from "
                  "precision_recall_curve; ties broken by the higher threshold; flag if score >= threshold",
          "data": "un-oversampled training split only (test set not used)",
          "n_rows": len(y), "n_positive": int(y.sum()), "n_splits": N_SPLITS, "cv_seed": SEED,
          "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)",
          "inside_each_fold": "bmi median and one-hot encoder fitted on fold training rows; RandomOverSampler "
                              "on fold training rows only; held-out rows not oversampled",
          "folds": fold_info, "runtime_seconds": time.perf_counter() - t0, "models": {}}
guard_failed = []
for name, scores in oof.items():
    prec, rec, thr = precision_recall_curve(y, scores)
    f2 = np.where((4 * prec[:-1] + rec[:-1]) > 0,
                  5 * prec[:-1] * rec[:-1] / np.maximum(4 * prec[:-1] + rec[:-1], 1e-12), 0.0)
    best = np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))
    i = best[np.argmax(thr[best])]                                  # tie -> higher threshold
    t = float(thr[i])
    flagged = scores >= t
    pred = flagged.astype(int)
    model_res = {
        "threshold": t,
        "oof_precision": float(precision_score(y, pred, zero_division=0)),
        "oof_recall": float(recall_score(y, pred)),
        "oof_f2": float(fbeta_score(y, pred, beta=2)),
        "oof_share_flagged": float(flagged.mean()),
        "oof_n_flagged": int(flagged.sum()),
        "oof_at_0_5": {"precision": float(precision_score(y, (scores >= 0.5).astype(int), zero_division=0)),
                       "recall": float(recall_score(y, (scores >= 0.5).astype(int))),
                       "f2": float(fbeta_score(y, (scores >= 0.5).astype(int), beta=2)),
                       "share_flagged": float((scores >= 0.5).mean())},
        "n_tied_best": int(len(best)),
        "fit_warnings": sorted(fit_warnings[name]),
    }
    assert abs(model_res["oof_f2"] - float(f2[i])) < 1e-9
    result["models"][name] = model_res
    if model_res["oof_share_flagged"] > MAX_FLAGGED_SHARE:
        guard_failed.append(name)

pd.DataFrame({"y_true": y, "oof_random_forest": oof["random_forest"], "oof_mlp": oof["mlp"]}) \
    .to_csv(TH / "oof_predictions.csv", index_label="train_row")
print(json.dumps({k: v for k, v in result["models"].items()}, indent=1))
if guard_failed:
    (TH / "thresholds_GUARD_FAILED.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.exit(f"Guard: threshold flags more than 50% of out-of-fold rows for {guard_failed}. Ask the user.")
(TH / "thresholds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
