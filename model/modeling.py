"""Shared model definitions and evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def build_models(feature_names: list[str]):
    """Return reproducible pipelines for every classifier required by the brief."""
    scaled_prep = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), feature_names),
        remainder="drop",
    )
    unscaled_prep = make_column_transformer(
        (SimpleImputer(strategy="median"), feature_names), remainder="drop"
    )
    return {
        "Logistic Regression": make_pipeline(
            scaled_prep, LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": make_pipeline(
            unscaled_prep,
            DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        ),
        "kNN": make_pipeline(scaled_prep, KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make_pipeline(scaled_prep, GaussianNB()),
        "Random Forest (Ensemble)": make_pipeline(
            unscaled_prep,
            RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
    }


def evaluate_classifier(model, X, y) -> dict[str, float]:
    """Calculate all six metrics requested in the assignment."""
    prediction = model.predict(X)
    average = "binary" if len(np.unique(y)) == 2 else "weighted"
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        auc = (
            roc_auc_score(y, probabilities[:, 1])
            if probabilities.shape[1] == 2
            else roc_auc_score(y, probabilities, multi_class="ovr", average="weighted")
        )
    else:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, prediction),
        "AUC": auc,
        "Precision": precision_score(y, prediction, average=average, zero_division=0),
        "Recall": recall_score(y, prediction, average=average, zero_division=0),
        "F1": f1_score(y, prediction, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }

