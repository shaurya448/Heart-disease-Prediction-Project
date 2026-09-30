"""Shared helpers: data loading, label maps, explanations, tips and PDF report."""
from datetime import datetime

import pandas as pd

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL

NICE = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "Vessels coloured", "thal": "Thalassemia",
}

# Friendly label -> code used in the dataset
CHOICES = {
    "sex": {"Male": 1, "Female": 0},
    "cp": {"Typical angina": 3, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 0},
    "fbs": {"No": 0, "Yes": 1},
    "restecg": {"Normal": 1, "ST-T wave abnormality": 2, "Left ventricular hypertrophy": 0},
    "exang": {"No": 0, "Yes": 1},
    "slope": {"Upsloping": 2, "Flat": 1, "Downsloping": 0},
    "thal": {"Normal": 2, "Fixed defect": 1, "Reversible defect": 3},
}
LABELS = {f: {v: k for k, v in m.items()} for f, m in CHOICES.items()}


def load_data(path="heart.csv"):
    """Load and clean the UCI heart disease data. Target: 1 = disease, 0 = no disease."""
    df = pd.read_csv(path).drop_duplicates()
    df = df[(df["ca"] <= 3) & (df["thal"] != 0)].copy()
    df["target"] = 1 - df["target"]  # in the raw file 1 means NO disease
    return df.reset_index(drop=True)


def risk_level(p):
    return "Low" if p < 0.30 else "Moderate" if p < 0.60 else "High"


def readable(row):
    """Feature -> human readable value."""
    return {NICE[f]: (LABELS[f][int(row[f])] if f in LABELS else f"{row[f]:g}") for f in FEATURES}


def explain(model, row, reference):
    """Local explanation: change in risk when each feature is replaced by a typical
    value of people WITHOUT heart disease. Positive = that value raises your risk."""
    x = pd.DataFrame([row])[FEATURES]
    base = model.predict_proba(x)[0, 1]
    variants = []
    for f in FEATURES:
        v = x.copy()
        v[f] = reference[f]
        variants.append(v)
    probs = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    out = pd.DataFrame({"Feature": [NICE[f] for f in FEATURES], "Effect": base - probs})
    return out.reindex(out["Effect"].abs().sort_values().index)


def flags(row):
    out = []
    if row["trestbps"] >= 140: out.append(f"Blood pressure is high ({row['trestbps']:g} mm Hg; normal is below 120/80)")
    if row["chol"] >= 240: out.append(f"Cholesterol is high ({row['chol']:g} mg/dl; desirable is below 200)")
    if row["fbs"] == 1: out.append("Fasting blood sugar is above 120 mg/dl")
    if row["exang"] == 1: out.append("Chest pain occurs during exercise")
    if row["oldpeak"] >= 2: out.append(f"ST depression is notable ({row['oldpeak']:g})")
    if row["ca"] >= 1: out.append(f"{int(row['ca'])} major vessel(s) show narrowing on fluoroscopy")
    if row["thalach"] < 0.7 * (220 - row["age"]): out.append(f"Maximum heart rate ({row['thalach']:g} bpm) is low for your age")
    return out


def tips(row):
    t = []
    if row["trestbps"] >= 130: t.append("Reduce salt and processed food and monitor your blood pressure regularly.")
    if row["chol"] >= 200: t.append("Cut down saturated and fried fats; add fibre, fruit and vegetables.")
    if row["fbs"] == 1: t.append("Keep blood sugar in check with less sugar and regular diabetes screening.")
    if row["exang"] == 1: t.append("Ask a doctor about safe exercise levels before intense activity.")
    if row["age"] >= 55: t.append("Have regular heart check-ups, as risk rises with age.")
    t.append("Aim for about 150 minutes of moderate activity a week, avoid smoking and limit alcohol.")
    return t


HEART_INFO = (
    "**What it is:** Heart disease refers to conditions that affect the heart, most commonly coronary artery "
    "disease, where arteries supplying the heart are narrowed by plaque. This can reduce blood flow and lead to "
    "chest pain or a heart attack.\n\n"
    "**Common symptoms:** chest pain or pressure, shortness of breath, unusual tiredness, pain spreading to the "
    "arm, neck or jaw, dizziness or palpitations.\n\n"
    "**Common risk factors:** high blood pressure, high cholesterol, diabetes, smoking, obesity, physical "
    "inactivity, family history and increasing age.\n\n"
    "**Seek emergency help immediately** if you have severe chest pain, trouble breathing or fainting."
)


def make_pdf(row, prob, risk, model_name, effects, flag_list):
    """Build a one-page PDF report and return it as bytes."""
    from fpdf import FPDF

    def t(s):  # core PDF fonts only support latin-1
        return str(s).replace("\u2265", ">=").encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    nl = dict(new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Heart Disease Risk Report", **nl)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, t(f"Generated {datetime.now():%d %b %Y, %H:%M}  |  Model: {model_name}"), **nl)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, t(f"Estimated likelihood: {prob:.0%}  ({risk} risk)"), **nl)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Details entered", **nl)
    pdf.set_font("Helvetica", "", 10)
    for k, v in readable(row).items():
        pdf.cell(70, 6, t(k), border=1)
        pdf.cell(0, 6, t(v), border=1, **nl)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Main factors affecting this result", **nl)
    pdf.set_font("Helvetica", "", 10)
    for _, r in effects.tail(5).iloc[::-1].iterrows():
        way = "raises" if r["Effect"] > 0 else "lowers"
        pdf.cell(0, 6, t(f"- {r['Feature']}: {way} risk by about {abs(r['Effect']) * 100:.0f} points"), **nl)
    if flag_list:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Values that stand out", **nl)
        pdf.set_font("Helvetica", "", 10)
        for f in flag_list:
            pdf.multi_cell(0, 6, t(f"- {f}"), **nl)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This is an educational prototype, not a medical diagnosis. It must not be used "
                   "to diagnose a condition or to guide treatment. Please consult a qualified doctor.", **nl)
    return bytes(pdf.output())


# ----------------------------------------------------------------- UI helpers
DEFAULTS = {"age": 50, "sex": "Male", "cp": "Typical angina", "trestbps": 120, "chol": 200, "thalach": 150,
            "oldpeak": 1.0, "ca": 0, "fbs": "No", "restecg": "Normal", "exang": "No", "slope": "Upsloping",
            "thal": "Normal"}

PRESETS = {
    "Low": {"age": 35, "sex": "Female", "cp": "Non-anginal pain", "trestbps": 118, "chol": 190, "thalach": 172,
            "oldpeak": 0.0, "ca": 0, "fbs": "No", "restecg": "Normal", "exang": "No", "slope": "Upsloping",
            "thal": "Normal"},
    "Moderate": {"age": 55, "sex": "Male", "cp": "Atypical angina", "trestbps": 135, "chol": 235, "thalach": 140,
                 "oldpeak": 1.5, "ca": 1, "fbs": "No", "restecg": "Normal", "exang": "No", "slope": "Flat",
                 "thal": "Normal"},
    "High": {"age": 63, "sex": "Male", "cp": "Asymptomatic", "trestbps": 150, "chol": 260, "thalach": 120,
             "oldpeak": 2.5, "ca": 2, "fbs": "Yes", "restecg": "ST-T wave abnormality", "exang": "Yes",
             "slope": "Flat", "thal": "Reversible defect"},
}


def bp_category(v):
    return ("🟢", "Normal") if v < 120 else ("🟡", "Elevated") if v < 130 else ("🟠", "High (stage 1)") if v < 140 else ("🔴", "High (stage 2)")


def chol_category(v):
    return ("🟢", "Desirable") if v < 200 else ("🟡", "Borderline high") if v < 240 else ("🔴", "High")


GLOSSARY = {
    "Angina": "Chest pain or pressure caused by reduced blood flow to the heart.",
    "Typical / atypical angina": "Typical angina follows the classic pattern (pressure with effort, relieved by rest); atypical only partly fits it.",
    "Non-anginal pain": "Chest pain that is probably not caused by the heart.",
    "Asymptomatic": "No chest pain. Heart disease can still be present without symptoms.",
    "Resting blood pressure": "Blood pressure at rest (systolic, in mm Hg). Below 120 is normal; 140 or more is high.",
    "Cholesterol": "A fatty substance in the blood. Total cholesterol below 200 mg/dl is desirable; 240 or more is high.",
    "Fasting blood sugar": "Blood sugar after not eating for 8+ hours. Above 120 mg/dl can point to diabetes risk.",
    "Resting ECG": "A recording of the heart's electrical activity at rest. Abnormal results include ST-T wave changes and thickened heart muscle (LV hypertrophy).",
    "Maximum heart rate": "Highest heart rate reached during an exercise test. A rough expected maximum is 220 minus your age.",
    "Exercise-induced angina": "Chest pain that appears during exercise.",
    "ST depression (oldpeak)": "How much the ST segment of the ECG drops during exercise compared with rest. Larger values are more concerning.",
    "ST slope": "Direction of the ST segment at peak exercise: upsloping is usually normal, flat or downsloping is more concerning.",
    "Vessels coloured (ca)": "Number of major heart vessels (0-3) showing narrowing on a fluoroscopy dye test.",
    "Thalassemia": "A blood-related test result: normal, fixed defect (permanent) or reversible defect (appears under stress).",
    "ROC-AUC": "How well a model separates people with and without disease. 0.5 is chance, 1.0 is perfect.",
    "Precision": "Of the people predicted to have disease, the share who really do.",
    "Recall": "Of the people who really have disease, the share the model catches.",
    "Decision threshold": "The likelihood above which the app calls a result 'higher likelihood'. Lower = fewer missed cases, more false alarms.",
}
