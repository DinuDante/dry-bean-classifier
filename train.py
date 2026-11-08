"""Reproduce the experiment. Run this file in BITS Virtual Lab for actual lab evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import platform
import urllib.request
import zipfile

import joblib
import pandas as pd
from scipy.io import arff
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_core import ROOT, FEATURES, TARGET, CLASSES, MODEL_FILES, evaluate, validate_frame

SEED = 5692
SOURCE_URL = "https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data():
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "dry_bean.csv"
    provenance_path = data_dir / "provenance.json"
    if csv_path.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if digest(csv_path) != provenance["csv_sha256"]:
            raise ValueError("Dataset hash changed. Restore the recorded original data.")
        return pd.read_csv(csv_path), provenance
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ML-coursework/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".arff")]
        if len(names) != 1:
            raise ValueError("Expected one UCI ARFF dataset.")
        contents = archive.read(names[0])
    records, _ = arff.loadarff(io.StringIO(contents.decode("utf-8-sig")))
    frame = pd.DataFrame(records)
    frame[TARGET] = frame[TARGET].map(lambda value: value.decode("utf-8"))
    if frame.shape != (13611, 17):
        raise ValueError(f"Unexpected UCI dataset shape: {frame.shape}")
    validate_frame(frame)
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    provenance = {
        "dataset": "UCI Dry Bean, dataset 602",
        "source_url": SOURCE_URL,
        "citation": "Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B",
        "license": "CC BY 4.0",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "arff_member": names[0],
        "arff_sha256": hashlib.sha256(contents).hexdigest(),
        "csv_sha256": digest(csv_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return frame, provenance


def candidates():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3, random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-9)),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(
            C=3.0, kernel="rbf", gamma="scale", probability=True, cache_size=256, random_state=SEED)),
    }


def run(execution_context="local"):
    original, provenance = prepare_data()
    original = validate_frame(original)
    conflicts = original.groupby(FEATURES, dropna=False)[TARGET].nunique()
    if (conflicts > 1).any():
        raise ValueError("Identical feature vectors have conflicting labels; review the source.")
    # Remove repeated measurements before splitting to prevent exact-feature leakage.
    clean = original.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    development, test = train_test_split(
        clean, test_size=0.20, stratify=clean[TARGET], random_state=SEED)
    train, validation = train_test_split(
        development, test_size=0.25, stratify=development[TARGET], random_state=SEED)
    train, validation, test = [part.sort_index() for part in (train, validation, test)]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "model").mkdir(exist_ok=True)
    test.to_csv(ROOT / "test_data.csv", index=False, lineterminator="\n")
    logs = []
    def log(message):
        print(message, flush=True)
        logs.append(message)
    log(f"Execution context (operator declared): {execution_context}; host platform: {platform.system()}")
    log(f"Original {len(original)}; removed duplicates {len(original)-len(clean)}; retained {len(clean)}")
    log(f"Train {len(train)}; validation {len(validation)}; held-out test {len(test)}; features {len(FEATURES)}")
    validation_rows = []
    definitions = candidates()
    for name, model in definitions.items():
        model.fit(train[FEATURES], train[TARGET])
        metrics, _ = evaluate(model, validation)
        validation_rows.append({"ML Model Name": name, **metrics})
        log(f"Validation {name}: macro F1={metrics['F1']:.6f}")
    val_table = pd.DataFrame(validation_rows)
    winner = str(val_table.loc[val_table["F1"].idxmax(), "ML Model Name"])
    log(f"Model selected using validation macro F1, before test evaluation: {winner}")
    test_rows, reports = [], {}
    for name, fitted in definitions.items():
        model = clone(fitted)
        # Refit on train + validation only. The held-out test is never fitted.
        model.fit(development[FEATURES], development[TARGET])
        joblib.dump(model, ROOT / "model" / MODEL_FILES[name], compress=3)
        metrics, details = evaluate(model, test)
        test_rows.append({"ML Model Name": name, **metrics})
        reports[name] = {key: details[key] for key in ("confusion_matrix", "classification_report")}
    table = pd.DataFrame(test_rows)
    table.to_csv(ROOT / "results" / "metrics.csv", index=False, lineterminator="\n")
    val_table.to_csv(ROOT / "results" / "validation_metrics.csv", index=False, lineterminator="\n")
    (ROOT / "results" / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    splits = {"train": train.index.tolist(), "validation": validation.index.tolist(), "test": test.index.tolist()}
    (ROOT / "results" / "split_indices.json").write_text(json.dumps(splits, indent=2) + "\n")
    summary = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_context_operator_declared": execution_context,
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "streamlit"]},
        "seed": SEED, "raw_rows": len(original), "duplicate_rows_removed": len(original)-len(clean),
        "usable_rows": len(clean), "feature_count": len(FEATURES),
        "split_rows": {key: len(value) for key, value in splits.items()},
        "class_counts_raw": {str(k): int(v) for k, v in original[TARGET].value_counts().items()},
        "class_counts_test": {str(k): int(v) for k, v in test[TARGET].value_counts().items()},
        "selected_model": winner, "selection_metric": "validation macro F1",
        "test_csv_sha256": digest(ROOT / "test_data.csv"),
        "source_csv_sha256": provenance["csv_sha256"],
        "model_sha256": {name: digest(ROOT / "model" / path) for name, path in MODEL_FILES.items()},
        "class_order": CLASSES,
    }
    (ROOT / "results" / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    log("COMPLETE: five saved pipelines and held-out metrics written. No uploaded CSV was used for training.")
    (ROOT / "results" / "training.log").write_text("\n".join(logs) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-context", choices=["local", "bits-lab"], default="local",
                        help="A descriptive label, not proof of where execution occurred.")
    args = parser.parse_args()
    run(args.execution_context)
