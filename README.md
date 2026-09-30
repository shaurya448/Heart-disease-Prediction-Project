# ❤️ Heart Disease Risk Predictor (Advanced Version)

A multi-tab machine learning web app that estimates the **likelihood of heart disease** from medical details, explains **why** the model gave that result, lets users try **what-if scenarios**, and exports a **PDF report**.

Built with **Python, Pandas, scikit-learn, Plotly, fpdf2 and Streamlit**. There is **no database and no separate backend**.

> ⚠️ **Disclaimer:** This is an **educational prototype, not a diagnostic tool**. Predictions must **not** be used to diagnose a condition or to guide treatment. Consult a qualified doctor for any health concern. For severe chest pain, trouble breathing or fainting, seek emergency help immediately.

---

## Table of Contents
1. [Features](#1-features)
2. [How It Works](#2-how-it-works)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Installation](#5-installation)
6. [Training the Models](#6-training-the-models)
7. [Running the App](#7-running-the-app)
8. [App Tour](#8-app-tour)
9. [Model Results](#9-model-results)
10. [Deploying Online](#10-deploying-online)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations](#12-limitations)
13. [Future Improvements](#13-future-improvements)

---

## 1. Features

| Area | What it does |
|---|---|
| **Live risk assessment** | Sliders and dropdowns grouped into 4 cards; the result (gauge, risk badge, likelihood, model agreement) **updates instantly** as you change any value, with no Predict button needed |
| **Quick-start examples** | One-click Low, Moderate and High-risk example profiles plus a Reset button |
| **Helpful hints** | Tooltips on every medical term, live blood pressure and cholesterol categories, expected max heart rate for your age |
| **Explainability** | "Why this result?" chart showing which of your values raise or lower the risk |
| **How you compare** | Radar chart against typical patients with and without disease, and your percentile among dataset patients |
| **What-if simulator + risk curve** | Adjust blood pressure, cholesterol, heart rate or blood sugar; plot how likelihood changes with any chosen factor |
| **Personalised guidance** | Values that stand out, heart disease information, tailored lifestyle tips, one-page **PDF report** |
| **Simple / detailed view** | Sidebar toggle hides the advanced analysis for a simpler screen |
| **Model insights** | Comparison table and bar chart with a metric selector, ROC curves with model picker, confusion matrix and threshold explorer, feature importance |
| **Data explorer** | Age and sex filters, distribution and scatter plots (choose any axes, "You" marker), correlation heatmap, data download |
| **Batch prediction** | Edit rows in an interactive table or upload a CSV; download results |
| **Session history** | Save results with a button, see a table, trend chart and change since the last saved result |
| **Glossary** | Searchable explanations of every medical and ML term used |
| **Settings sidebar** | Choose the prediction model and the decision threshold |

## 2. How It Works

```
heart.csv (UCI dataset)
   │  Pandas: clean, remove invalid rows, fix label
   ▼
Preprocessing (scikit-learn Pipeline): StandardScaler + OneHotEncoder
   ▼
Tune 4 models with GridSearchCV (5-fold CV):
Logistic Regression · Random Forest · SVM · Gradient Boosting
   ▼
Soft-voting Ensemble of the tuned models
   ▼
Evaluate on held-out 20% test set  →  choose best by cross-validated ROC-AUC
   ▼
Permutation importance + reference profile  →  saved with Joblib (model_artifacts.pkl)
   ▼
Streamlit app: prediction · explanation · what-if · report · insights · batch
```

**Key design choices**
- **Best model is chosen on cross-validation, not on test data**, so the test scores stay honest.
- **Explanations work with any model:** each of your values is replaced by the typical value of a person *without* heart disease, and the change in likelihood is measured.
- **The preprocessing is inside the saved pipeline**, so the app passes raw form values and gets identical scaling and encoding to training.

## 3. Project Structure

```
heart_project/
├── app.py               # Streamlit app (5 tabs, live interactive UI)
├── train.py             # Tunes, evaluates and saves the models
├── utils.py             # Shared helpers: data loading, label maps, explanations, tips, PDF report
├── heart.csv            # UCI Heart Disease dataset (303 records)
├── requirements.txt     # Python libraries
├── README.md
├── docs/                # SRS, PRD, synopsis and Agile sprint plan
└── model_artifacts.pkl  # (generated) models, metrics, ROC data, importance
```
If `model_artifacts.pkl` is missing, the app trains automatically on first launch (about a minute).

## 4. Dataset

- **Source:** UCI Machine Learning Repository, Heart Disease (Cleveland), 14-column version (https://archive.ics.uci.edu/dataset/45/heart+disease), CSV obtained from a public GitHub mirror.
- **Size:** 303 raw records → **296 after cleaning** (136 disease, 160 no disease)

| Column | Meaning | Values |
|---|---|---|
| `age` | Age in years | 29–77 |
| `sex` | Sex | 1 = male, 0 = female |
| `cp` | Chest pain type | 0 = asymptomatic, 1 = atypical angina, 2 = non-anginal, 3 = typical angina |
| `trestbps` | Resting blood pressure (mm Hg) | 94–200 |
| `chol` | Cholesterol (mg/dl) | 126–564 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = yes, 0 = no |
| `restecg` | Resting ECG | 0 = LV hypertrophy, 1 = normal, 2 = ST-T abnormality |
| `thalach` | Maximum heart rate achieved | 71–202 |
| `exang` | Exercise-induced angina | 1 = yes, 0 = no |
| `oldpeak` | ST depression | 0–6.2 |
| `slope` | Slope of peak exercise ST segment | 0 = downsloping, 1 = flat, 2 = upsloping |
| `ca` | Major vessels coloured (fluoroscopy) | 0–3 |
| `thal` | Thalassemia | 1 = fixed defect, 2 = normal, 3 = reversible defect |
| `target` | Outcome | see below |

**Cleaning (in `utils.load_data`):** drop the duplicate row; drop rows with invalid values (`ca` = 4, `thal` = 0); flip the label because in this CSV `target = 1` means *no* disease. After cleaning, **1 = disease, 0 = no disease**.

**Note on category codes:** the CSV has no codebook. The meanings of `cp`, `restecg`, `slope` and `thal` were inferred from the original UCI data and from disease rates in each category. Verify them if your guide provides an official codebook.

## 5. Installation

**Requirements:** Python 3.9+ (tick "Add Python to PATH" on Windows), a modern browser, about 500 MB free disk space.

```bash
# 1. Open a terminal in the project folder (Windows: type cmd in the folder's address bar)

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install the libraries
pip install -r requirements.txt
```
`requirements.txt` needs **Streamlit 1.50 or newer**.

## 6. Training the Models

```bash
python train.py
```
This takes under a minute on most laptops and prints a comparison table. It:
1. Loads and cleans the data
2. Splits 80% train / 20% test (stratified, `random_state=42`)
3. Tunes each model with `GridSearchCV` (5-fold, scored by ROC-AUC)
4. Builds a soft-voting ensemble of the tuned models
5. Evaluates all five on the test set (accuracy, precision, recall, F1, ROC-AUC)
6. Picks the best model by **cross-validated** ROC-AUC
7. Computes permutation importance and a "typical no-disease" reference profile
8. Saves everything to `model_artifacts.pkl`

**Customizing:** edit the `SEARCH` dictionary in `train.py` to change models or hyperparameter grids; change `test_size` or `random_state` in `train_test_split`; change the `sort_values("CV ROC-AUC")` line to choose by another metric (for example `"Recall"`). Delete `model_artifacts.pkl` or re-run `python train.py` after any change, then restart the app.

## 7. Running the App

```bash
streamlit run app.py
```
Open **http://localhost:8501** (it opens automatically). Stop with `Ctrl + C`.

## 8. App Tour

**🩺 Risk Assessment**
1. Optionally click an example profile (🟢 🟡 🔴) or **Reset**.
2. Set your values in the four cards: *About you*, *Vitals & blood tests*, *Symptoms*, *Heart tests*.
3. The right side updates live: risk badge, gauge, likelihood, risk level, number of models agreeing, and values that stand out.
4. Click **💾 Save this result to history** to keep it (a toast confirms).
5. Explore the sub-tabs: **Why this result**, **How you compare**, **What-if & risk curve**, **Tips & report** (PDF download).
6. Turn off **Show detailed analysis** in the sidebar for a simpler screen.

**📊 Model Insights** – Cross-validated and test metrics, a metric selector for the bar chart, ROC curves for chosen models, a confusion matrix that follows the sidebar **decision threshold**, and overall feature importance.

**🔍 Data Explorer** – Filter by age and sex, view distributions or a scatter plot with any two numeric features, see the correlation heatmap, and download the data.

**📁 Batch & History** – Edit a table of patients (add rows with the + at the bottom) or upload a CSV; download predictions. Below, the session history with a trend chart.

**ℹ️ About & Glossary** – Method, limitations, and a searchable glossary.

### Sample inputs
Use the example buttons, or enter: age 63, male, asymptomatic chest pain, exercise angina Yes, reversible defect, 2 vessels for a high-risk case (about 95%); age 35, female, non-anginal pain for a low-risk case (under 10%).

## 9. Model Results

Tuned with 5-fold CV on the training set; test set = about 60 patients.

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.900 ± 0.035 | 0.833 | 0.875 | 0.750 | 0.808 | 0.961 |
| Random Forest | 0.898 ± 0.024 | 0.867 | 0.885 | 0.821 | 0.852 | 0.958 |
| **SVM (selected)** | **0.906 ± 0.022** | 0.883 | 0.920 | 0.821 | 0.868 | 0.960 |
| Gradient Boosting | 0.885 ± 0.026 | 0.867 | 0.857 | 0.857 | 0.857 | 0.943 |
| Voting Ensemble | 0.903 ± 0.031 | 0.867 | 0.885 | 0.821 | 0.852 | 0.960 |

Your numbers may differ slightly with other library versions. Cross-validated ROC-AUC (about 0.90) is a more realistic estimate than the test ROC-AUC (about 0.96), because the test set is small.

**Metric guide:** *precision* = of those predicted as disease, how many truly have it; *recall* = of those who have disease, how many were caught; *ROC-AUC* = how well the model separates the two groups (1.0 is perfect).

## 10. Deploying Online

Deploy free on **Streamlit Community Cloud**:
1. Create a **public** GitHub repository.
2. Upload `app.py`, `train.py`, `utils.py`, `heart.csv` and `requirements.txt` (not `model_artifacts.pkl`).
3. On share.streamlit.io choose **Create app**, select the repository, branch `main`, main file `app.py`, then **Deploy**.
4. Wait a few minutes; the first start trains the models on the server.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` not recognized | `python -m streamlit run app.py` |
| `pip` not recognized | `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` again |
| `FileNotFoundError: heart.csv` | Run commands from the folder that contains `heart.csv` |
| Error about `width="stretch"` or other Streamlit arguments | Upgrade: `pip install -U streamlit` |
| Error loading `model_artifacts.pkl` after upgrading scikit-learn | Delete it and run `python train.py` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| PDF button fails | `pip install -U fpdf2` |
| **Windows blocks a `.pyd` file (Smart App Control)** | Turn off Smart App Control (Windows Security → App & browser control), or use WSL, or use Google Colab / Streamlit Cloud |

## 12. Limitations
- Small dataset (about 300 patients) from a single source and time period
- Only 13 inputs; smoking, weight and family history are not included
- Category codes in the CSV are inferred, not officially documented
- Explanations show statistical patterns, not medical causes; the what-if simulator is hypothetical
- Not clinically validated; may be less accurate for under-represented groups

## 13. Future Improvements
- User accounts and stored history (SQLite / MongoDB)
- SHAP-based explanations and probability calibration
- Larger combined datasets and external validation
- React frontend with a Flask / FastAPI backend
- Multilingual interface

## Acknowledgements
UCI Machine Learning Repository, Heart Disease dataset (Janosi, Steinbrunn, Pfisterer, Detrano); scikit-learn, Pandas, Plotly, fpdf2 and Streamlit communities.

## License and Use
For educational use. The app is **not a medical device**.
