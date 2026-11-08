"""Shared schema, saved pipelines and evaluation for the Dry Bean experiment."""
import csv
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent
TARGET = "Class"
FEATURES = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
            "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
            "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
            "ShapeFactor3", "ShapeFactor4"]
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MAX_UPLOAD_ROWS = 5000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def validate_frame(frame, max_rows=None):
    """Reject malformed input; never silently drop features or infer labels."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    expected = FEATURES + [TARGET]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"CSV columns do not match. Missing: {missing}. Extra: {extra}.")
    if frame.empty:
        raise ValueError("The CSV must contain at least one labelled test row.")
    if max_rows and len(frame) > max_rows:
        raise ValueError(f"Upload only test data, with at most {max_rows} rows.")
    if frame[TARGET].isna().any():
        raise ValueError("Class contains missing labels.")
    invalid = sorted(set(frame[TARGET].astype(str)) - set(CLASSES))
    if invalid:
        raise ValueError(f"Unknown Class labels: {invalid}. Use the original uppercase labels.")
    try:
        features = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All 16 feature columns must contain numeric values.") from exc
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature values must be finite; missing values and infinity are not allowed.")
    clean = features.copy()
    clean[TARGET] = frame[TARGET].astype(str).to_numpy()
    return clean


def read_test_csv(payload):
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The test CSV must be no larger than 5 MiB.")
    try:
        decoded = payload.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)))
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not allowed.")
        frame = pd.read_csv(io.StringIO(decoded))
    except (UnicodeDecodeError, StopIteration, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise ValueError("Upload a valid, non-empty UTF-8 CSV file.") from exc
    return validate_frame(frame, MAX_UPLOAD_ROWS)


def load_models():
    # Only trusted project artifacts are loaded. The UI accepts CSV, never pickle uploads.
    return {name: joblib.load(ROOT / "model" / filename)
            for name, filename in MODEL_FILES.items()}


def evaluate(model, frame):
    """Macro P/R/F1 across seven labels; macro one-vs-rest AUC from probabilities."""
    y = frame[TARGET]
    prediction = model.predict(frame[FEATURES])
    probability = model.predict_proba(frame[FEATURES])
    order = [list(model.classes_).index(label) for label in CLASSES]
    probability = probability[:, order]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, labels=CLASSES, average="macro", zero_division=0)
    # A fixed seven-class AUC needs positive and negative examples for every class.
    auc = (float(roc_auc_score(y, probability, labels=CLASSES,
                             multi_class="ovr", average="macro"))
           if set(y) == set(CLASSES) else None)
    metrics = {"Accuracy": float(accuracy_score(y, prediction)), "AUC": auc,
               "Precision": float(precision), "Recall": float(recall),
               "F1": float(f1), "MCC": float(matthews_corrcoef(y, prediction))}
    details = {
        "confusion_matrix": confusion_matrix(y, prediction, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y, prediction, labels=CLASSES, output_dict=True, zero_division=0),
        "predictions": prediction.tolist(),
        "probabilities": probability.tolist(),
    }
    return metrics, details
