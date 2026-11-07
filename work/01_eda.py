"""Task 1: data-quality facts and the EDA panel (Figure 1)."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_PATH, OUT, load_raw

EDA = OUT / "eda"
EDA.mkdir(parents=True, exist_ok=True)
C0, C1 = "#2A78D6", "#D1612A"   # no stroke, stroke
INK, MUTED = "#1F2933", "#616E7C"

df = load_raw()
n = len(df)
facts = {
    "rows": n,
    "columns": df.shape[1],
    "file_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_rows_ignoring_id": int(df.drop(columns="id").duplicated().sum()),
    "duplicate_ids": int(df["id"].duplicated().sum()),
}

# Missingness, with "Unknown" smoking reported as unavailable information
miss = pd.DataFrame({"missing_n": df.isna().sum()})
miss.loc["smoking_status (Unknown)"] = int((df["smoking_status"] == "Unknown").sum())
miss["missing_pct"] = (miss["missing_n"] / n * 100).round(2)
miss.to_csv(EDA / "missing_values.csv", index_label="column")

counts = df["stroke"].value_counts().sort_index()
facts["stroke_0"] = int(counts[0])
facts["stroke_1"] = int(counts[1])
facts["stroke_pct_positive"] = round(counts[1] / n * 100, 2)

rows = []
for col in ["age", "avg_glucose_level", "bmi"]:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    rows.append({"variable": col, "n": len(s), "mean": s.mean(), "sd": s.std(),
                 "median": s.median(), "q1": q1, "q3": q3, "iqr": q3 - q1,
                 "min": s.min(), "max": s.max()})
numeric = pd.DataFrame(rows).round(2)
numeric.to_csv(EDA / "numeric_summary.csv", index=False)

cat_rows = []
for col in ["gender", "hypertension", "heart_disease", "ever_married", "work_type",
            "Residence_type", "smoking_status"]:
    for level, c in df[col].value_counts().items():
        rate = df.loc[df[col] == level, "stroke"].mean() * 100
        cat_rows.append({"variable": col, "level": level, "n": int(c),
                         "pct": round(c / n * 100, 2), "stroke_rate_pct": round(rate, 2)})
cats = pd.DataFrame(cat_rows)
cats.to_csv(EDA / "categorical_summary.csv", index=False)

# Plausibility flags
facts.update({
    "age_under_1": int((df["age"] < 1).sum()),
    "age_non_integer": int((df["age"] % 1 != 0).sum()),
    "age_under_18": int((df["age"] < 18).sum()),
    "bmi_under_12": int((df["bmi"] < 12).sum()),
    "bmi_over_60": int((df["bmi"] > 60).sum()),
    "gender_other": int((df["gender"] == "Other").sum()),
    "glucose_over_200": int((df["avg_glucose_level"] > 200).sum()),
})

# Comparisons by stroke status
by = df.groupby("stroke")
facts["median_age_by_stroke"] = {int(k): float(v) for k, v in by["age"].median().items()}
facts["median_glucose_by_stroke"] = {int(k): float(v) for k, v in by["avg_glucose_level"].median().items()}
facts["median_bmi_by_stroke"] = {int(k): float(v) for k, v in by["bmi"].median().items()}
facts["pct_age_ge_60_within_stroke1"] = round((df.loc[df.stroke == 1, "age"] >= 60).mean() * 100, 2)
facts["pct_age_ge_60_within_stroke0"] = round((df.loc[df.stroke == 0, "age"] >= 60).mean() * 100, 2)
facts["stroke_rate_bmi_missing_pct"] = round(df.loc[df.bmi.isna(), "stroke"].mean() * 100, 2)
facts["stroke_rate_bmi_present_pct"] = round(df.loc[df.bmi.notna(), "stroke"].mean() * 100, 2)
facts["stroke1_with_bmi_missing"] = int(df.loc[df.bmi.isna(), "stroke"].sum())
(EDA / "eda_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

# ---- Figure 1 ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
fig, axes = plt.subplots(3, 3, figsize=(13, 11.5))
ax = axes.ravel()

bars = ax[0].bar(["No stroke (0)", "Stroke (1)"], [counts[0], counts[1]],
                 color=[C0, C1], width=0.55)
for b, v in zip(bars, [counts[0], counts[1]]):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 70, f"{v} ({v / n * 100:.1f}%)", ha="center")
ax[0].set_ylim(0, counts[0] * 1.13)
ax[0].set_ylabel("Patients")
ax[0].set_title("A. Stroke class counts")

labels = ["bmi\n(blank in file)", "smoking_status\n(\"Unknown\")"]
vals = [int(df.bmi.isna().sum()), int((df.smoking_status == "Unknown").sum())]
bars = ax[1].bar(labels, vals, color=C0, width=0.55)
for b, v in zip(bars, vals):
    ax[1].text(b.get_x() + b.get_width() / 2, v + 25, f"{v} ({v / n * 100:.1f}%)", ha="center")
ax[1].set_ylim(0, max(vals) * 1.15)
ax[1].set_ylabel("Patients")
ax[1].set_title("B. Missing or unavailable values")

ax[2].axis("off")
ax[2].set_title("Key", loc="left")
ax[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C0), plt.Rectangle((0, 0), 1, 1, color=C1)],
             labels=["No stroke (0), n=4861", "Stroke (1), n=249"], loc="upper left",
             frameon=False, fontsize=11)
ax[2].text(0.0, 0.62, "Histograms show the share of each\nclass, so the two groups can be\n"
                      "compared despite the imbalance.\nAll 5110 rows; associations only.",
           transform=ax[2].transAxes, va="top", color=MUTED, fontsize=10.5)

for a, col, title, unit in [(ax[3], "age", "C. Age", "Age (years)"),
                            (ax[4], "avg_glucose_level", "D. Average glucose", "avg_glucose_level (unit not stated in source)"),
                            (ax[5], "bmi", "E. BMI (observed values)", "bmi (kg/m²)")]:
    data = df[[col, "stroke"]].dropna()
    bins = np.histogram_bin_edges(data[col], bins=30)
    for s, c, lab in [(0, C0, "No stroke"), (1, C1, "Stroke")]:
        vals_ = data.loc[data.stroke == s, col]
        a.hist(vals_, bins=bins, weights=np.ones(len(vals_)) / len(vals_) * 100,
               color=c, alpha=0.65, edgecolor="white", linewidth=0.4, label=lab)
    a.set_xlabel(unit)
    a.set_ylabel("% of class")
    a.set_title(title)

for a, col, title, ylabel in [(ax[6], "age", "F. Age by stroke status", "Age (years)"),
                              (ax[7], "avg_glucose_level", "G. Glucose by stroke status", "avg_glucose_level")]:
    box = a.boxplot([df.loc[df.stroke == s, col] for s in (0, 1)], widths=0.5, patch_artist=True,
                    tick_labels=["No stroke", "Stroke"], flierprops=dict(marker=".", markersize=3,
                    markerfacecolor=MUTED, markeredgecolor=MUTED),
                    medianprops=dict(color="white", linewidth=2))
    for patch, c in zip(box["boxes"], [C0, C1]):
        patch.set(facecolor=c, edgecolor=c)
    a.set_ylabel(ylabel)
    a.set_title(title)

rates = [("Hypertension\n= 0", df[df.hypertension == 0].stroke.mean()),
         ("Hypertension\n= 1", df[df.hypertension == 1].stroke.mean()),
         ("Heart disease\n= 0", df[df.heart_disease == 0].stroke.mean()),
         ("Heart disease\n= 1", df[df.heart_disease == 1].stroke.mean())]
vals = [r[1] * 100 for r in rates]
bars = ax[8].bar([r[0] for r in rates], vals, color=C1, width=0.6)
for b, v in zip(bars, vals):
    ax[8].text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center")
ax[8].set_ylim(0, max(vals) * 1.15)
ax[8].set_ylabel("Patients with stroke = 1 (%)")
ax[8].tick_params(axis="x", labelsize=9)
ax[8].set_title("H. Stroke share by condition")

fig.tight_layout(h_pad=2.2, w_pad=2)
fig.savefig(EDA / "figure1_eda_panel.png", dpi=170, facecolor="white")
print(json.dumps(facts, indent=1))
print(miss[miss.missing_n > 0])
print(numeric.to_string())
print(cats.to_string())
