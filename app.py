"""Interactive evaluation of five saved classifiers; this app never trains models."""
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from ml_core import ROOT, FEATURES, TARGET, CLASSES, METRICS, evaluate, load_models, read_test_csv

st.set_page_config(page_title="Dry Bean | Model Comparison", page_icon="🫘", layout="wide")
st.title("Dry Bean Classification")
st.caption("Machine Learning - Assignment 2 | DINESH BEHERA | 2026AJ05692")
st.write("Compare five classifiers using 16 measured shape features from seven dry bean varieties.")


@st.cache_resource
def saved_models():
    return load_models()


@st.cache_data(max_entries=8, show_spinner=False)
def score_csv(payload):
    frame = read_test_csv(payload)
    rows, details = [], {}
    for name, model in saved_models().items():
        scores, detail = evaluate(model, frame)
        rows.append({"ML Model Name": name, **scores})
        details[name] = detail
    return frame, pd.DataFrame(rows), details


try:
    models = saved_models()
    summary = json.loads((ROOT / "results" / "execution_summary.json").read_text())
except (OSError, ValueError, ImportError) as exc:
    st.error(f"Could not load the saved experiment. Run train.py with the pinned requirements. {exc}")
    st.stop()

with st.sidebar:
    st.header("Test data")
    source = st.radio("Choose evaluation data", ["Bundled held-out test data", "Upload test CSV"])
    st.download_button("Download test_data.csv", (ROOT / "test_data.csv").read_bytes(),
                       "test_data.csv", "text/csv")
    uploaded = st.file_uploader("Upload labelled test CSV", type=["csv"])
    st.caption("Required: all 16 feature columns and Class. Maximum 5,000 rows / 5 MiB.")
    chosen = st.selectbox("Classifier", list(models),
                          index=list(models).index(summary["selected_model"]))
    st.caption("Saved preprocessing is applied automatically. Uploaded rows are evaluated only.")

if source == "Upload test CSV":
    if uploaded is None:
        st.info("Upload a labelled test CSV, or select the bundled held-out test data.")
        st.stop()
    payload = uploaded.getvalue()
else:
    payload = (ROOT / "test_data.csv").read_bytes()

try:
    with st.spinner("Evaluating the saved classifiers..."):
        frame, comparison, all_details = score_csv(payload)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

is_benchmark = hashlib.sha256(payload).hexdigest() == summary["test_csv_sha256"]
st.subheader("Evaluation data")
st.write(f"{len(frame):,} rows · {len(FEATURES)} features · {frame[TARGET].nunique()} of 7 classes")
if is_benchmark:
    st.caption("This is the untouched test split used in the README comparison.")
else:
    st.caption("These results describe your uploaded data. They do not replace the held-out benchmark.")
if frame[TARGET].nunique() != len(CLASSES):
    st.warning("AUC is N/A because one or more of the seven classes is absent. "
               "Macro precision, recall and F1 still include all seven classes; undefined class scores are zero.")

st.subheader(chosen)
selected = comparison.set_index("ML Model Name").loc[chosen]
for column, metric in zip(st.columns(6), METRICS):
    value = selected[metric]
    column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
st.caption("Precision / Recall / F1: macro averages. AUC: macro one-vs-rest. MCC: multiclass.")
st.subheader("All five models on these test rows")
st.dataframe(comparison.set_index("ML Model Name").style.format("{:.4f}", na_rep="N/A"), width="stretch")
st.download_button("Download comparison CSV", comparison.to_csv(index=False),
                   "model_comparison.csv", "text/csv")

matrix_tab, report_tab, rows_tab = st.tabs(["Confusion matrix", "Classification report", "Data and predictions"])
details = all_details[chosen]
with matrix_tab:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(np.array(details["confusion_matrix"]), display_labels=CLASSES).plot(
        ax=ax, cmap="Greys", colorbar=False, xticks_rotation=45, values_format="d")
    ax.set_title(chosen)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Rows: actual class. Columns: predicted class.")
with report_tab:
    report = pd.DataFrame(details["classification_report"]).transpose()
    st.dataframe(report.style.format("{:.4f}"), width="stretch")
with rows_tab:
    predictions = frame.copy()
    predictions["PredictedClass"] = details["predictions"]
    st.dataframe(predictions, width="stretch", hide_index=True)
    st.download_button("Download predictions", predictions.to_csv(index=False),
                       "predictions.csv", "text/csv")
with st.expander("Dataset and experimental method"):
    st.write("UCI Dry Bean: 13,611 original records, 16 numeric features, seven classes. "
             "Exact duplicate feature vectors are removed before a stratified 60/20/20 "
             "train/validation/test split. The validation macro F1 selects the model. "
             "All five models are refitted on train + validation and evaluated on the same held-out test rows.")
    st.write(f"Validation-selected model: {summary['selected_model']}. Random seed: {summary['seed']}.")
    st.caption("Column spellings AspectRation and roundness follow the original UCI file.")
    st.link_button("UCI dataset and attribution", "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset")
