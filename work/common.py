"""Shared, leak-safe data pipeline for the stroke assignment.

Differences from the course utils.py (which is left unchanged):
  * the split happens first and is stratified on stroke
  * bmi is imputed with the median of the training split, not with 0
  * the one-hot encoder is fitted on the training split only
  * oversampling touches the training split only and is seeded
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 2021
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "outputs"

TARGET = "stroke"
ID_COL = "id"
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
NUMERIC = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]


def load_raw():
    return pd.read_csv(DATA_PATH)


def fit_preprocessing(X_raw_train):
    """Fit the bmi median and one-hot encoder on the given training rows only.

    Returns (transform, bmi_median, encoder, feature_names). Used for the final
    split and inside each cross-validation training fold.
    """
    bmi_median = float(X_raw_train["bmi"].median())
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    encoder.fit(X_raw_train[CATEGORICAL].astype(object))
    feature_names = NUMERIC + list(encoder.get_feature_names_out(CATEGORICAL))

    def transform(frame):
        num = frame[NUMERIC].astype(float).copy()
        num["bmi"] = num["bmi"].fillna(bmi_median)
        cat = encoder.transform(frame[CATEGORICAL].astype(object))
        return pd.DataFrame(
            np.hstack([num.to_numpy(), cat]), columns=feature_names, index=frame.index
        )

    return transform, bmi_median, encoder, feature_names


def oversample(X, y):
    """RandomOverSampler on training rows only, seeded."""
    sampler = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
    return sampler.fit_resample(X, y)


def prepare_data():
    """Return a dict with the split, encoded features and fitted preprocessing."""
    raw = load_raw()
    X = raw[NUMERIC + CATEGORICAL].copy()   # id and stroke are excluded here
    y = raw[TARGET].astype(int)

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    transform, bmi_median, encoder, feature_names = fit_preprocessing(X_tr_raw)
    X_train = transform(X_tr_raw)
    X_test = transform(X_te_raw)
    X_res, y_res = oversample(X_train, y_train)

    return {
        "raw": raw,
        "X_train_raw": X_tr_raw,
        "X_test_raw": X_te_raw,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_res": X_res,
        "y_train_res": y_res,
        "bmi_median": bmi_median,
        "encoder": encoder,
        "feature_names": feature_names,
    }


def binary_metrics(y_true, y_pred, y_score, threshold=0.5):
    """Test metrics asked for by the report template."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_stroke1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_stroke1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_stroke1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_support": int(np.sum(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "prevalence_test": float(np.mean(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
