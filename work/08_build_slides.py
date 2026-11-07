"""Task 5: exactly three plain slides, using the same case and numbers as the report."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from common import OUT

SL = OUT / "slides"
SL.mkdir(parents=True, exist_ok=True)
FONT = "Calibri"
INK = RGBColor(0x26, 0x26, 0x26)
GREY = RGBColor(0x59, 0x59, 0x59)
ACCENT, NEUTRAL = "#C0504D", "#7F7F7F"
SOURCE = "Dataset: Kaggle stroke prediction, fedesoriano. Student project, not a clinical tool."


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


m = load("task2/metrics.json")
split = load("task2/split_info.json")
case = load("task3/selected_case.json")

by_feature = {r["feature"]: r["shap"] for r in case["top3"]}
parts = [("Heart disease on your record", by_feature["heart_disease"]),
         ("Your average blood sugar", by_feature["avg_glucose_level"]),
         ("Other recorded details together", case["remaining_features_sum"]),
         ("Your body mass index (BMI)", by_feature["bmi"])]
assert parts[0][1] > 0 and parts[1][1] > 0 and parts[2][1] > 0 and parts[3][1] < 0

plt.rcParams.update({"font.family": FONT, "font.size": 18})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
labels = [p[0] for p in parts][::-1]
values = [p[1] for p in parts][::-1]
ax.barh(labels, values, color=[ACCENT if v > 0 else NEUTRAL for v in values], height=0.55)
ax.axvline(0, color="#262626", linewidth=1)
ax.set_xticks([])
ax.tick_params(axis="y", length=0, labelsize=17)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)
lim = max(abs(v) for v in values) * 1.05
ax.set_xlim(-lim, lim)
ax.text(0.47, -0.05, "pushed down", transform=ax.transAxes, ha="right", va="top", fontsize=16, color="#595959")
ax.text(0.53, -0.05, "pushed up", transform=ax.transAxes, ha="left", va="top", fontsize=16, color="#595959")
fig.tight_layout()
fig.savefig(SL / "patient_contributions.png", dpi=200, facecolor="white")
plt.close(fig)

# The force plot note repeats the bullets, so it is cropped off for the slide to make the plot larger
force = Image.open(OUT / "task3/force_plot_case.png")
force.crop((0, 0, force.width, int(force.height * 0.70))).save(SL / "force_plot_for_slide.png")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
LAYOUT = prs.slide_layouts[1]   # Title and Content


def add_slide(title, bullets, body_box, picture, picture_box, alt, size=20):
    slide = prs.slides.add_slide(LAYOUT)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.8)
    t.text_frame.text = title
    t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    par = t.text_frame.paragraphs[0]
    par.alignment = PP_ALIGN.LEFT
    par.runs[0].font.size, par.runs[0].font.bold = Pt(32), True
    par.runs[0].font.name, par.runs[0].font.color.rgb = FONT, INK

    body = slide.placeholders[1]
    body.left, body.top, body.width, body.height = [Inches(v) for v in body_box]
    frame = body.text_frame
    frame.word_wrap = True
    for i, text in enumerate(bullets):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(9)
        run = p.add_run()
        run.text = text
        run.font.size, run.font.name, run.font.color.rgb = Pt(size), FONT, INK

    x, y, w, h = picture_box
    iw, ih = Image.open(picture).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    pic = slide.shapes.add_picture(str(picture), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                   Inches(pw), Inches(ph))
    pic.name = "Plot"
    pic._element.nvPicPr.cNvPr.set("descr", alt)

    note = slide.shapes.add_textbox(Inches(0.6), Inches(6.95), Inches(12.1), Inches(0.35))
    note.text_frame.text = SOURCE
    run = note.text_frame.paragraphs[0].runs[0]
    run.font.size, run.font.name, run.font.color.rgb = Pt(12), FONT, GREY
    return slide


add_slide(
    "Model performance and explanation of stroke classification",
    [f"RandomForest, 80/20 stratified split, only the train set oversampled. Positive class is stroke = 1",
     f"Test set: ROC AUC {m['roc_auc']:.3f}, AP {m['average_precision']:.3f}, recall "
     f"{m['recall_stroke1']:.3f}, precision {m['precision_stroke1']:.3f} (threshold 0.5)",
     "Top features are age, avg_glucose_level and bmi",
     "TreeExplainer, tree path dependent, so no background data. Output is P(stroke = 1)",
     f"Limitation: trained on balanced data, so scores are not calibrated (base value {case['base_value']:.2f}, "
     f"test prevalence {m['prevalence_test'] * 100:.2f}%)"],
    (6.9, 1.25, 5.85, 5.6), OUT / "task4/summary_plot.png", (0.5, 1.2, 6.2, 5.7),
    "SHAP summary plot for the RandomForest on the held-out test set. Age has the widest spread.", size=19)

fn, pos = m["fn"], m["positive_support"]
add_slide(
    "Interpreting one patient prediction in clinical context",
    [f"Predicted probability of stroke = 1 is {case['predicted_probability']:.2f}, from a base value of "
     f"{case['base_value']:.2f}. Actual class was 1",
     "Heart disease and high average glucose push it up. BMI of 36.6 pushes it down",
     f"The model missed {fn} of {pos} recorded strokes at threshold 0.5, so a low score does not rule out a stroke history",
     "Calibration was not checked and there is no external validation yet",
     "SHAP explains how the model behaves. It is not a diagnosis"],
    (0.6, 3.85, 12.1, 3.1), SL / "force_plot_for_slide.png", (0.9, 1.1, 11.5, 2.7),
    "SHAP force plot for one test case, from base value 0.50 to output 0.72.", size=19)

add_slide(
    "What the computer prediction means for this example",
    ["The computer looked at recorded details such as your age, health conditions, blood sugar, BMI and smoking status",
     "In this example, heart disease and your blood sugar pushed the result up. Your BMI pushed it down",
     "This only shows how the computer worked. It does not show what caused anything, and it is not health advice",
     "The prediction can be wrong. It learned from old records and it misses many people",
     "A clinician reads this together with your other information"],
    (7.0, 1.3, 5.75, 5.5), SL / "patient_contributions.png", (0.5, 1.5, 6.4, 4.6),
    "Bar chart in plain words: heart disease, blood sugar and other details pushed the result up; BMI pushed it down.", size=20)

assert len(prs.slides) == 3
prs.save(SL / "Stroke_SHAP_Slides.pptx")
print("saved", len(prs.slides), "slides")
