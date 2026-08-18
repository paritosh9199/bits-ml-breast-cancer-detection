"""Streamlit evaluation dashboard for ML Assignment 2."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.modeling import evaluate_classifier


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model" / "artifacts"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

st.set_page_config(page_title="Cancer Classifier Lab", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
      background:#f8fafc;
      border:1px solid #cbd5e1;
      padding:12px;
      border-radius:12px;
      box-shadow:0 1px 3px rgba(15, 23, 42, 0.08);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
      color:#475569 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
      color:#0f172a !important;
    }
    .eyebrow {color:#0f766e; font-weight:700; letter-spacing:.12em; font-size:.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    metadata = joblib.load(ARTIFACT_DIR / "metadata.joblib")
    models = {
        name: joblib.load(ARTIFACT_DIR / filename) for name, filename in MODEL_FILES.items()
    }
    return metadata, models


st.markdown('<br><br><p class="eyebrow">BITS WILP · MACHINE LEARNING · PARITOSH SRIVASTAVA · 2025AC05189</p>', unsafe_allow_html=True)
st.title("Breast Cancer Classification Lab")
st.caption("Compare five classifiers on labelled diagnostic test data.")

if not (ARTIFACT_DIR / "metadata.joblib").exists():
    st.error("Model artifacts are missing. Run `python train_models.py` and restart the app.")
    st.stop()

metadata, models = load_assets()

with st.sidebar:
    st.header("Experiment controls")
    selected_name = st.selectbox("Classification model", list(models))
    uploaded_file = st.file_uploader("Upload labelled test data", type="csv")
    st.caption("Expected target column: `target` (0 = malignant, 1 = benign).")
    with st.expander("Required CSV columns"):
        st.write(metadata["feature_names"] + [metadata["target_name"]])

data = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(ROOT / "test_data.csv")
source = "uploaded CSV" if uploaded_file else "included test_data.csv"
required = set(metadata["feature_names"] + [metadata["target_name"]])
missing = sorted(required - set(data.columns))
if missing:
    st.error(f"The CSV is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

X_test = data[metadata["feature_names"]]
y_test = data[metadata["target_name"]]
model = models[selected_name]
prediction = model.predict(X_test)
metrics = evaluate_classifier(model, X_test, y_test)

st.subheader(selected_name)
st.caption(f"Evaluated {len(data):,} labelled rows from {source}.")
metric_columns = st.columns(6)
for column, (label, value) in zip(metric_columns, metrics.items()):
    column.metric(label, f"{value:.3f}")

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Confusion matrix")
    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, prediction, labels=labels)
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="crest",
        cbar=False,
        xticklabels=[metadata["target_labels"].get(int(v), str(v)) for v in labels],
        yticklabels=[metadata["target_labels"].get(int(v), str(v)) for v in labels],
        ax=ax,
    )
    ax.set(xlabel="Predicted label", ylabel="Actual label")
    fig.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_test,
        prediction,
        labels=labels,
        target_names=[metadata["target_labels"].get(int(v), str(v)) for v in labels],
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T
    st.dataframe(report_frame.style.format("{:.3f}"), width="stretch")

with st.expander("Preview evaluated data"):
    preview = data.copy()
    preview["predicted_target"] = prediction
    st.dataframe(preview.head(25), width="stretch")
