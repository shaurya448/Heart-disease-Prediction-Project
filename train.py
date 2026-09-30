"""Train, tune and evaluate models for heart disease prediction.

Run:  python train.py
Output: model_artifacts.pkl (models, metrics, ROC curves, confusion matrices, feature importance)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from utils import CATEGORICAL, FEATURES, NUMERIC, load_data

ARTIFACT = "model_artifacts.pkl"

# Model + hyperparameter grid (tuned with 5-fold cross-validation on the training set)
SEARCH = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=42),
                      {"model__n_estimators": [100, 300], "model__max_depth": [3, 5, None],
                       "model__min_samples_leaf": [1, 3]}),
    "SVM": (SVC(probability=True, random_state=42), {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1],
                           "model__max_depth": [2, 3]}),
}


def make_pipe(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def main():
    df = load_data()
    X, y = df[FEATURES], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    # 1) Tune each model with GridSearchCV
    models, cv_scores = {}, {}
    for name, (est, grid) in SEARCH.items():
        print(f"Tuning {name} ...")
        gs = GridSearchCV(make_pipe(est), grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
        models[name] = gs.best_estimator_
        cv_scores[name] = (gs.best_score_, gs.cv_results_["std_test_score"][gs.best_index_])

    # 2) Soft-voting ensemble of the tuned models
    print("Training Voting Ensemble ...")
    ens = VotingClassifier([(n.replace(" ", "_"), m) for n, m in models.items()], voting="soft")
    s = cross_val_score(ens, X_tr, y_tr, cv=cv, scoring="roc_auc")
    models["Voting Ensemble"] = ens.fit(X_tr, y_tr)
    cv_scores["Voting Ensemble"] = (s.mean(), s.std())

    # 3) Evaluate on the untouched 20% test set
    rows, roc, cms, probs = [], {}, {}, {}
    for name, m in models.items():
        p = m.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, p)
        roc[name], probs[name] = (fpr, tpr), p
        cms[name] = confusion_matrix(y_te, pred)
        rows.append({
            "Model": name, "CV ROC-AUC": cv_scores[name][0], "CV Std": cv_scores[name][1],
            "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred), "Test ROC-AUC": roc_auc_score(y_te, p),
        })
    metrics = pd.DataFrame(rows).round(3)
    best = metrics.sort_values("CV ROC-AUC", ascending=False).iloc[0]["Model"]  # chosen on CV, not on test data
    print("\n" + metrics.to_string(index=False))
    print(f"\nBest model (highest cross-validated ROC-AUC): {best}")

    # 4) Global feature importance (permutation importance on test set)
    pi = permutation_importance(models[best], X_te, y_te, scoring="roc_auc", n_repeats=30, random_state=42)
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean, "Std": pi.importances_std})
    importance = importance.sort_values("Importance")

    # 5) Reference profile = typical values of patients WITHOUT disease (used for explanations)
    healthy = X_tr[y_tr == 0]
    reference = {f: float(healthy[f].median()) for f in NUMERIC}
    reference.update({f: int(healthy[f].mode()[0]) for f in CATEGORICAL})

    joblib.dump({
        "models": models, "best": best, "metrics": metrics, "roc": roc, "cms": cms,
        "y_test": y_te.to_numpy(), "probs": probs, "importance": importance, "reference": reference,
        "n_train": len(X_tr), "n_test": len(X_te),
    }, ARTIFACT)
    print(f"Saved {ARTIFACT}")


if __name__ == "__main__":
    main()
