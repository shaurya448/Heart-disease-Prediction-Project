"""Heart Disease Risk Predictor - interactive multi-tab Streamlit app (educational prototype).

Run:  streamlit run app.py
"""
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import train
import utils
from utils import CHOICES, DEFAULTS, FEATURES, GLOSSARY, LABELS, NICE, NUMERIC, PRESETS, risk_level

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.hero{padding:1.3rem 1.6rem;border-radius:16px;margin-bottom:1rem;color:#fff;
      background:linear-gradient(120deg,#8e0000,#d32f2f 55%,#ff7043)}
.hero h1{color:#fff;margin:0;padding:0;font-size:2rem}
.hero p{margin:.35rem 0 0;opacity:.95;font-size:1.02rem}
.badge{display:inline-block;padding:.4rem 1rem;border-radius:999px;color:#fff;font-weight:700;
       letter-spacing:.04em;font-size:1.05rem}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:12px;
       padding:.6rem .9rem;background:rgba(128,128,128,.06)}
button[kind="primary"]{font-weight:600}
</style>
""", unsafe_allow_html=True)

COLORS = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#c62828"}
OUTCOME_COLORS = {"Disease": "#c62828", "No disease": "#2e7d32"}


# ------------------------------------------------------------------ data + models
@st.cache_resource(show_spinner="Training models for the first time (about a minute)...")
def load_artifacts():
    if not os.path.exists(train.ARTIFACT):
        train.main()
    return joblib.load(train.ARTIFACT)


@st.cache_data
def load_df():
    return utils.load_data()


art, df = load_artifacts(), load_df()
names = list(art["models"])
ss = st.session_state
for k, v in DEFAULTS.items():
    ss.setdefault(f"in_{k}", v)
ss.setdefault("history", [])


def apply_values(values):  # used by the preset / reset buttons
    for k, v in values.items():
        ss[f"in_{k}"] = v


def current_row():
    return {f: (CHOICES[f][ss[f"in_{f}"]] if f in CHOICES else ss[f"in_{f}"]) for f in FEATURES}


def predict(m, rows):
    return m.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Prediction model", names, index=names.index(art["best"]),
                              help="The default has the best cross-validated ROC-AUC.")
    threshold = st.slider("Decision threshold", 0.20, 0.80, 0.50, 0.05,
                          help="Results at or above this likelihood are called 'higher likelihood'. "
                               "Lower it to catch more possible cases (more false alarms).")
    detailed = st.toggle("Show detailed analysis", value=True,
                         help="Turn off for a simpler view with only the result, tips and report.")
    st.divider()
    st.caption(f"Trained on {art['n_train']} patients, tested on {art['n_test']}.")
    st.caption("Educational prototype. Not a medical device.")
model = art["models"][model_name]

st.markdown("""
<div class="hero"><h1>❤️ Heart Disease Risk Predictor</h1>
<p>Enter your details, see an instant estimate, find out why, and explore what could change it.</p></div>
""", unsafe_allow_html=True)
st.warning("**Educational prototype only.** This is not a medical diagnosis and must not be used to diagnose "
           "or decide treatment. Please consult a doctor for any health concern.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🩺 Risk Assessment", "📊 Model Insights", "🔍 Data Explorer", "📁 Batch & History", "ℹ️ About & Glossary"])

# ================================================================== TAB 1
with tab1:
    with st.expander("👋 How to use this app"):
        st.markdown("1. **Enter your details** on the left, or load an example profile.\n"
                    "2. **Watch the result update instantly** on the right.\n"
                    "3. Explore **why** the model gave that result, **how you compare**, and **what-if** changes.\n"
                    "4. **Save** the result to your history or **download a PDF report**.")

    b1, b2, b3, b4, _ = st.columns([1, 1, 1, 0.8, 1.6])
    b1.button("🟢 Low-risk example", on_click=apply_values, args=(PRESETS["Low"],), width="stretch")
    b2.button("🟡 Moderate example", on_click=apply_values, args=(PRESETS["Moderate"],), width="stretch")
    b3.button("🔴 High-risk example", on_click=apply_values, args=(PRESETS["High"],), width="stretch")
    b4.button("↺ Reset", on_click=apply_values, args=(DEFAULTS,), width="stretch")

    left, right = st.columns([1, 1.25], gap="large")

    # ---------------- input panel
    with left:
        with st.container(border=True):
            st.markdown("#### 👤 About you")
            c1, c2 = st.columns(2)
            c1.slider("Age (years)", 20, 100, key="in_age")
            c2.selectbox("Sex", list(CHOICES["sex"]), key="in_sex")
        with st.container(border=True):
            st.markdown("#### 🩸 Vitals & blood tests")
            bp = st.slider("Resting blood pressure (mm Hg)", 80, 220, key="in_trestbps",
                           help="The top (systolic) number, measured at rest.")
            st.caption("{} {}".format(*utils.bp_category(bp)))
            ch = st.slider("Cholesterol (mg/dl)", 100, 600, key="in_chol", help="Total serum cholesterol.")
            st.caption("{} {}".format(*utils.chol_category(ch)))
            hr = st.slider("Maximum heart rate reached (bpm)", 60, 220, key="in_thalach",
                           help="Highest heart rate during an exercise test.")
            exp_hr = 220 - ss["in_age"]
            st.caption(f"Rough expected maximum for your age: {exp_hr} bpm (you: {hr / exp_hr:.0%})")
            st.selectbox("Fasting blood sugar above 120 mg/dl?", list(CHOICES["fbs"]), key="in_fbs")
        with st.container(border=True):
            st.markdown("#### 🫀 Symptoms")
            s1, s2 = st.columns(2)
            s1.selectbox("Chest pain type", list(CHOICES["cp"]), key="in_cp",
                         help="Typical angina: pressure with effort, relieved by rest. Asymptomatic: no chest pain.")
            s2.selectbox("Chest pain during exercise?", list(CHOICES["exang"]), key="in_exang")
        with st.container(border=True):
            st.markdown("#### 🧪 Heart tests")
            t1, t2 = st.columns(2)
            t1.selectbox("Resting ECG result", list(CHOICES["restecg"]), key="in_restecg")
            t2.selectbox("Thalassemia test", list(CHOICES["thal"]), key="in_thal")
            t1.slider("ST depression (oldpeak)", 0.0, 7.0, step=0.1, key="in_oldpeak",
                      help="ECG change during exercise compared with rest.")
            t2.selectbox("ST slope at peak exercise", list(CHOICES["slope"]), key="in_slope")
            st.select_slider("Major vessels coloured (0-3)", [0, 1, 2, 3], key="in_ca",
                             help="Vessels showing narrowing on a fluoroscopy dye test.")

    # ---------------- live result
    row = current_row()
    prob = float(predict(model, [row])[0])
    risk = risk_level(prob)
    allp = {n: float(predict(m, [row])[0]) for n, m in art["models"].items()}
    agree = sum(p >= threshold for p in allp.values())
    eff = utils.explain(model, row, art["reference"])
    flag_list = utils.flags(row)

    with right:
        with st.container(border=True):
            st.markdown(f'<span class="badge" style="background:{COLORS[risk]}">{risk.upper()} RISK</span>',
                        unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[risk]},
                       "steps": [{"range": [0, 30], "color": "rgba(46,125,50,.18)"},
                                 {"range": [30, 60], "color": "rgba(249,168,37,.20)"},
                                 {"range": [60, 100], "color": "rgba(198,40,40,.18)"}],
                       "threshold": {"line": {"color": "gray", "width": 3}, "value": threshold * 100}}))
            fig.update_layout(height=250, margin=dict(t=30, b=0, l=30, r=30))
            st.plotly_chart(fig, width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("Likelihood", f"{prob:.0%}")
            m2.metric("Risk level", risk)
            m3.metric("Models agreeing", f"{agree}/{len(allp)}")
            if prob >= threshold:
                st.error("**Higher likelihood of heart disease** for the values entered. Please see a doctor "
                         "or cardiologist for proper tests and advice.")
            else:
                st.success("**Lower likelihood of heart disease** for the values entered. This does not "
                           "guarantee a healthy heart; regular check-ups still matter.")
            if flag_list:
                st.markdown("**Values that stand out:**")
                for f in flag_list:
                    st.markdown(f"- {f}")
            if st.button("💾 Save this result to history", type="primary", width="stretch"):
                ss.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"), "Model": model_name, "Age": row["age"],
                    "Sex": ss["in_sex"], "BP": row["trestbps"], "Cholesterol": row["chol"],
                    "Max HR": row["thalach"], "Likelihood (%)": round(prob * 100, 1), "Risk": risk})
                st.toast(f"Saved to history ({len(ss.history)} total)", icon="✅")

    # ---------------- detailed analysis
    st.divider()
    if detailed:
        d1, d2, d3, d4 = st.tabs(["🧠 Why this result", "👥 How you compare", "🎛️ What-if & risk curve", "📋 Tips & report"])
    else:
        (d4,) = st.tabs(["📋 Tips & report"])

    if detailed:
        with d1:
            g1, g2 = st.columns([1.4, 1])
            with g1:
                bar = go.Figure(go.Bar(x=eff["Effect"] * 100, y=eff["Feature"], orientation="h",
                                       marker_color=["#c62828" if v > 0 else "#2e7d32" for v in eff["Effect"]]))
                bar.update_layout(height=430, margin=dict(t=10, b=10), xaxis_title="Effect on likelihood (percentage points)")
                st.plotly_chart(bar, width="stretch")
                st.caption("Each bar shows how much your likelihood would change if that value were replaced by the "
                           "typical value of a person without heart disease. 🔴 raises risk, 🟢 lowers it.")
            with g2:
                st.markdown("**All models compared**")
                st.dataframe(pd.DataFrame({"Model": list(allp), "Likelihood (%)": [p * 100 for p in allp.values()]}),
                             hide_index=True, width="stretch",
                             column_config={"Likelihood (%)": st.column_config.ProgressColumn(
                                 format="%.0f%%", min_value=0, max_value=100)})
                st.caption("If models disagree, the estimate is less certain.")

        with d2:
            r1, r2 = st.columns([1.2, 1])
            with r1:
                num = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
                lo, hi = df[num].min(), df[num].max()
                norm = lambda s: ((pd.Series(s)[num] - lo) / (hi - lo)).clip(0, 1).tolist()
                radar = go.Figure()
                for label, vals, col in [("You", norm(row), "#1565c0"),
                                         ("Typical: no disease", norm(df[df.target == 0][num].median()), "#2e7d32"),
                                         ("Typical: disease", norm(df[df.target == 1][num].median()), "#c62828")]:
                    radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[NICE[f] for f in num] + [NICE[num[0]]],
                                                    name=label, fill="toself" if label == "You" else "none",
                                                    line=dict(color=col, width=3 if label == "You" else 2), opacity=0.9))
                radar.update_layout(height=420, margin=dict(t=30, b=30), polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)))
                st.plotly_chart(radar, width="stretch")
                st.caption("Values scaled 0-1 across the dataset. Compare your shape with the typical patient with and without disease.")
            with r2:
                st.markdown("**Where you stand among the 296 patients in the dataset**")
                for f, lab in [("age", "Age"), ("trestbps", "Blood pressure"), ("chol", "Cholesterol"), ("thalach", "Max heart rate")]:
                    pct = float((df[f] < row[f]).mean())
                    st.markdown(f"{lab}: higher than **{pct:.0%}** of patients")
                    st.progress(pct)

        with d3:
            st.markdown("**What-if simulator**")
            st.caption("Adjust the factors you can influence and see how the estimate changes. Hypothetical only.")
            w1, w2, w3, w4 = st.columns(4)
            dbp = w1.slider("Blood pressure change", -50, 50, 0, help="mm Hg, relative to your current value")
            dch = w2.slider("Cholesterol change", -150, 150, 0, help="mg/dl, relative to your current value")
            dhr = w3.slider("Max heart rate change", -40, 40, 0, help="bpm, relative to your current value")
            fb0 = w4.toggle("Blood sugar controlled (below 120)", value=False, disabled=row["fbs"] == 0)
            sim = {**row, "trestbps": int(np.clip(row["trestbps"] + dbp, 80, 220)),
                   "chol": int(np.clip(row["chol"] + dch, 100, 600)),
                   "thalach": int(np.clip(row["thalach"] + dhr, 60, 220)),
                   "fbs": 0 if fb0 else row["fbs"]}
            p_sim = float(predict(model, [sim])[0])
            k1, k2, k3 = st.columns(3)
            k1.metric("Current likelihood", f"{prob:.0%}")
            k2.metric("Simulated likelihood", f"{p_sim:.0%}", f"{(p_sim - prob) * 100:+.0f} points", delta_color="inverse")
            k3.metric("Simulated values", f"{sim['trestbps']} / {sim['chol']} / {sim['thalach']}", "BP / chol / max HR", delta_color="off")

            st.markdown("**Risk curve**")
            sweep_f = st.selectbox("See how likelihood changes with…", ["chol", "trestbps", "thalach", "age", "oldpeak"],
                                   format_func=lambda f: NICE[f])
            grid = np.linspace(df[sweep_f].min(), df[sweep_f].max(), 40)
            rows = [{**row, sweep_f: g} for g in grid]
            curve = go.Figure()
            curve.add_trace(go.Scatter(x=grid, y=predict(model, rows) * 100, mode="lines", name="Likelihood", line=dict(width=3, color="#c62828")))
            curve.add_trace(go.Scatter(x=[row[sweep_f]], y=[prob * 100], mode="markers", name="You",
                                       marker=dict(size=13, color="#1565c0", line=dict(width=2, color="white"))))
            curve.add_hline(y=threshold * 100, line_dash="dash", line_color="gray", annotation_text="threshold")
            curve.update_layout(height=340, margin=dict(t=10), xaxis_title=NICE[sweep_f], yaxis_title="Likelihood (%)", yaxis_range=[0, 100])
            st.plotly_chart(curve, width="stretch")
            st.caption("Other inputs are held at your current values.")

    with d4:
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Personalised lifestyle tips**")
            for t in utils.tips(row):
                st.markdown(f"- {t}")
        with e2:
            with st.expander("About heart disease", expanded=prob >= threshold):
                st.markdown(utils.HEART_INFO)
        st.download_button("📄 Download PDF report", utils.make_pdf(row, prob, risk, model_name, eff, flag_list),
                           file_name="heart_risk_report.pdf", mime="application/pdf")

# ================================================================== TAB 2
with tab2:
    st.subheader("Model comparison")
    st.caption("Models were tuned with 5-fold cross-validation on the training set and then tested once on held-out data. "
               "The default model is chosen by cross-validated ROC-AUC, not by test results.")
    st.dataframe(art["metrics"], hide_index=True, width="stretch")

    metric = st.radio("Compare models by", ["CV ROC-AUC", "Test ROC-AUC", "Accuracy", "Precision", "Recall", "F1"], horizontal=True)
    cmp = px.bar(art["metrics"], x="Model", y=metric, color="Model", text_auto=".3f", range_y=[0, 1])
    cmp.update_layout(height=320, margin=dict(t=10), showlegend=False)
    st.plotly_chart(cmp, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC curves (test set)**")
        shown = st.multiselect("Models to show", names, default=names)
        roc = go.Figure()
        for n in shown:
            fpr, tpr = art["roc"][n]
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n, line=dict(width=4 if n == model_name else 2)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey")))
        roc.update_layout(height=380, xaxis_title="False positive rate", yaxis_title="True positive rate", margin=dict(t=10))
        st.plotly_chart(roc, width="stretch")
    with c2:
        st.markdown(f"**Threshold explorer: {model_name}**")
        yt, pt = art["y_test"], art["probs"][model_name]
        pred = (pt >= threshold).astype(int)
        tp, tn = int(((pred == 1) & (yt == 1)).sum()), int(((pred == 0) & (yt == 0)).sum())
        fp, fn = int(((pred == 1) & (yt == 0)).sum()), int(((pred == 0) & (yt == 1)).sum())
        cm = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                       x=["Predicted: No disease", "Predicted: Disease"], y=["Actual: No disease", "Actual: Disease"])
        cm.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(cm, width="stretch")
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{(tp + tn) / len(yt):.0%}")
        b.metric("Precision", f"{tp / max(tp + fp, 1):.0%}")
        c.metric("Recall", f"{tp / max(tp + fn, 1):.0%}")
        st.caption(f"At threshold {threshold:.2f}: {fn} disease case(s) missed, {fp} false alarm(s). Move the sidebar slider to see the trade-off.")

    st.markdown(f"**Which inputs matter most overall** (permutation importance of {art['best']})")
    imp = art["importance"].copy()
    imp["Feature"] = imp["Feature"].map(NICE)
    fi = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h",
                          error_x=dict(type="data", array=imp["Std"]), marker_color="#1F6B4F"))
    fi.update_layout(height=420, margin=dict(t=10), xaxis_title="Drop in ROC-AUC when the feature is shuffled")
    st.plotly_chart(fi, width="stretch")

# ================================================================== TAB 3
with tab3:
    st.subheader("Explore the training data")
    with st.expander("🎚️ Filter the data"):
        f1, f2 = st.columns(2)
        amin, amax = int(df.age.min()), int(df.age.max())
        age_rng = f1.slider("Age range", amin, amax, (amin, amax))
        sexes = f2.multiselect("Sex", list(CHOICES["sex"]), default=list(CHOICES["sex"]))
    dfx = df[df.age.between(*age_rng) & df.sex.isin([CHOICES["sex"][s] for s in sexes])]

    if dfx.empty:
        st.warning("No patients match the filter. Widen the age range or select a sex.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Patients", len(dfx))
        d2.metric("With disease", f"{dfx.target.mean():.0%}")
        d3.metric("Average age", f"{dfx.age.mean():.0f}")
        d4.metric("Features", len(FEATURES))
        plot_df = dfx.assign(Outcome=dfx.target.map({1: "Disease", 0: "No disease"}))
        row = current_row()

        v1, v2 = st.tabs(["Distribution", "Scatter plot"])
        with v1:
            feat = st.selectbox("Feature", FEATURES, format_func=lambda f: NICE[f])
            if feat in LABELS:
                pdx = plot_df.assign(**{feat: plot_df[feat].map(LABELS[feat])})
                g = px.histogram(pdx, x=feat, color="Outcome", barmode="group", color_discrete_map=OUTCOME_COLORS)
            else:
                g = px.histogram(plot_df, x=feat, color="Outcome", barmode="overlay", opacity=0.65, marginal="box",
                                 color_discrete_map=OUTCOME_COLORS)
                g.add_vline(x=row[feat], line_dash="dash", line_color="black", annotation_text="You")
            g.update_layout(height=380, margin=dict(t=10), xaxis_title=NICE[feat])
            st.plotly_chart(g, width="stretch")
        with v2:
            s1, s2 = st.columns(2)
            xf = s1.selectbox("X axis", NUMERIC, index=0, format_func=lambda f: NICE[f])
            yf = s2.selectbox("Y axis", NUMERIC, index=3, format_func=lambda f: NICE[f])
            sc = px.scatter(plot_df, x=xf, y=yf, color="Outcome", opacity=0.7, color_discrete_map=OUTCOME_COLORS)
            sc.add_trace(go.Scatter(x=[row[xf]], y=[row[yf]], mode="markers", name="You",
                                    marker=dict(symbol="star", size=18, color="#1565c0", line=dict(width=2, color="white"))))
            sc.update_layout(height=400, margin=dict(t=10), xaxis_title=NICE[xf], yaxis_title=NICE[yf])
            st.plotly_chart(sc, width="stretch")

        st.markdown("**Correlation heatmap** (target = 1 for disease)")
        corr = px.imshow(dfx.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        corr.update_layout(height=560, margin=dict(t=10))
        st.plotly_chart(corr, width="stretch")
        with st.expander("View data table"):
            st.dataframe(dfx, width="stretch")
            st.download_button("Download data (CSV)", dfx.to_csv(index=False), "heart_cleaned.csv", "text/csv")

# ================================================================== TAB 4
with tab4:
    st.subheader("Batch prediction")
    u1, u2 = st.tabs(["✏️ Type or edit rows", "📤 Upload CSV"])

    with u1:
        st.caption("Edit the table directly, add rows with the + at the bottom, then see predictions below.")
        base = pd.DataFrame([PRESETS["Low"], PRESETS["Moderate"], PRESETS["High"]])[FEATURES].rename(columns=NICE)
        cfg = {}
        for f in FEATURES:
            if f in CHOICES:
                cfg[NICE[f]] = st.column_config.SelectboxColumn(NICE[f], options=list(CHOICES[f]), required=True)
            elif f == "ca":
                cfg[NICE[f]] = st.column_config.SelectboxColumn(NICE[f], options=[0, 1, 2, 3], required=True)
            else:
                lo, hi = float(df[f].min()), float(df[f].max())
                cfg[NICE[f]] = st.column_config.NumberColumn(NICE[f], min_value=lo * 0.7, max_value=hi * 1.3, required=True)
        edited = st.data_editor(base, num_rows="dynamic", column_config=cfg, width="stretch", key="editor")
        ed = edited.rename(columns={v: k for k, v in NICE.items()}).dropna()
        if len(ed):
            for f in CHOICES:
                ed[f] = ed[f].map(CHOICES[f])
            ed = ed.dropna()
            if len(ed):
                p = predict(model, ed.to_dict("records"))
                res = pd.DataFrame({"Row": range(1, len(ed) + 1), "Likelihood (%)": (p * 100).round(1),
                                    "Risk": [risk_level(x) for x in p],
                                    "Prediction": np.where(p >= threshold, "Higher likelihood", "Lower likelihood")})
                st.dataframe(res, hide_index=True, width="stretch",
                             column_config={"Likelihood (%)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})

    with u2:
        st.caption("Upload a CSV with the 13 feature columns using dataset codes. Download the template to see the format.")
        st.download_button("Download CSV template", df[FEATURES].head(5).to_csv(index=False), "template.csv", "text/csv")
        up = st.file_uploader("Upload CSV", type="csv")
        if up is not None:
            try:
                data = pd.read_csv(up)
                missing = [c for c in FEATURES if c not in data.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    data = data[FEATURES].apply(pd.to_numeric, errors="coerce")
                    bad = int(data.isna().any(axis=1).sum())
                    data = data.dropna()
                    if bad:
                        st.warning(f"{bad} row(s) with missing or non-numeric values were skipped.")
                    p = model.predict_proba(data)[:, 1]
                    out = data.assign(**{"Likelihood (%)": (p * 100).round(1), "Risk": [risk_level(x) for x in p],
                                         "Prediction": np.where(p >= threshold, "Higher likelihood", "Lower likelihood")})
                    st.dataframe(out, width="stretch")
                    st.bar_chart(out["Risk"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0))
                    st.download_button("Download results (CSV)", out.to_csv(index=False), "predictions.csv", "text/csv")
            except Exception as e:  # unreadable file
                st.error(f"Could not read the file: {e}")

    st.divider()
    st.subheader("Session history")
    st.caption("Results you saved in this browser session. Nothing is stored on a server or in a database.")
    if ss.history:
        hist = pd.DataFrame(ss.history)
        st.dataframe(hist, hide_index=True, width="stretch",
                     column_config={"Likelihood (%)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
        if len(hist) >= 2:
            delta = hist["Likelihood (%)"].iloc[-1] - hist["Likelihood (%)"].iloc[-2]
            st.metric("Change since previous saved result", f"{hist['Likelihood (%)'].iloc[-1]:.1f}%", f"{delta:+.1f} points", delta_color="inverse")
        st.line_chart(hist.set_index(hist.index + 1)["Likelihood (%)"])
        h1, h2 = st.columns(2)
        h1.download_button("Download history (CSV)", hist.to_csv(index=False), "history.csv", "text/csv")
        if h2.button("Clear history"):
            ss.history = []
            st.rerun()
    else:
        st.info("No saved results yet. Use **Save this result to history** in the Risk Assessment tab.")

# ================================================================== TAB 5
with tab5:
    a1, a2 = st.columns([1.2, 1], gap="large")
    with a1:
        st.subheader("About this project")
        st.markdown(f"""
**Purpose:** an educational demonstration of an end-to-end machine learning application for heart disease risk.

**Data:** UCI Heart Disease (Cleveland), {len(df)} cleaned patient records, 13 features. Duplicates and invalid
values were removed and the label was set so that 1 = disease.

**Pipeline:** Pandas cleaning → scikit-learn preprocessing (scaling + one-hot encoding) → 4 models tuned with
GridSearchCV (5-fold) → soft-voting ensemble → evaluation on a held-out 20% test set → best model chosen by
cross-validated ROC-AUC → saved with Joblib.

**Explanations:** the "Why this result?" chart replaces each of your values with a typical value from people without
heart disease and measures the change in likelihood. It is an approximation that works with any model.

**Limitations**
- Small dataset (about 300 patients) from one source and time period; test scores are uncertain.
- Only 13 inputs; smoking, weight and family history are not included.
- Not clinically validated. Explanations show statistical patterns, not causes.
- The what-if simulator and risk curve are hypothetical and are not medical advice.

**Disclaimer:** predictions must not be used to diagnose a condition or to guide treatment. Seek emergency help
for severe chest pain, trouble breathing or fainting.
""")
    with a2:
        st.subheader("📖 Glossary")
        q = st.text_input("Search a term", placeholder="e.g. ECG, cholesterol, recall")
        hits = {k: v for k, v in GLOSSARY.items() if q.lower() in k.lower() or q.lower() in v.lower()}
        if not hits:
            st.info("No matching terms.")
        for k, v in hits.items():
            with st.expander(k):
                st.write(v)
