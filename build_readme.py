"""Generate the required README from measured results, never from invented scores."""
import argparse
import json
from urllib.parse import urlparse

import pandas as pd
from ml_core import ROOT, FEATURES, METRICS


def github_url(value):
    url = urlparse(value)
    if url.scheme != "https" or url.hostname != "github.com" or len(url.path.strip("/").split("/")) != 2:
        raise ValueError("Use the actual HTTPS GitHub repository URL (owner/repository).")
    return value.rstrip("/")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def build(repository_url=None):
    summary = json.loads((ROOT / "results/execution_summary.json").read_text())
    scores = pd.read_csv(ROOT / "results/metrics.csv").set_index("ML Model Name")
    val = pd.read_csv(ROOT / "results/validation_metrics.csv").set_index("ML Model Name")
    reports = json.loads((ROOT / "results/reports.json").read_text())
    rows = [[name] + [f"{scores.loc[name, col]:.4f}" for col in METRICS] for name in scores.index]
    mechanisms = {
        "Logistic Regression": "A regularized linear boundary provides a useful baseline after standardization; remaining errors are consistent with overlapping class shapes.",
        "Decision Tree": "The depth-12 tree captures nonlinear feature thresholds without scaling. Its axis-aligned splits can miss smoother boundaries; depth and leaf limits restrict overfitting.",
        "KNN": "Distance-weighted voting with seven neighbors benefits from scaling because raw area and shape ratios have very different magnitudes.",
        "Naive Bayes": "Gaussian conditional independence is restrictive here because area, convex area and axis lengths are related. This is a plausible explanation, not a causal test.",
        "Support Vector Machine": "An RBF kernel models nonlinear boundaries in standardized feature space. Probabilities are fitted internally on development data, without the held-out test.",
    }
    observations=[]
    for name in scores.index:
        report=reports[name]["classification_report"]
        weakest=min(summary["class_order"], key=lambda label: report[label]["recall"])
        observations.append([name, f"Lowest class recall: {weakest} ({report[weakest]['recall']:.4f}). " + mechanisms[name]])
    winner=summary["selected_model"]
    top_test=scores["F1"].idxmax()
    observations.append(["Overall winner", f"{winner}, selected by validation macro F1 ({val.loc[winner,'F1']:.4f}) before test evaluation. Its test macro F1 is {scores.loc[winner,'F1']:.4f}. The highest test macro F1 in this run belongs to {top_test} ({scores.loc[top_test,'F1']:.4f}). These are results of one fixed split, not a statistical superiority claim."])
    counts=", ".join(f"{name}: {count}" for name,count in summary["class_counts_raw"].items())
    link=(f"[GitHub repository]({github_url(repository_url)})" if repository_url else
          "A separate evaluator-accessible assignment repository/link is pending. This project may be backed up inside the private coursework repository; that backup does not establish evaluator access or deployment.")
    text=f"""# Machine Learning Assignment 2 - Dry Bean Classification

DINESH BEHERA | Student | BITS ID: 2026AJ05692 | AIML ZG565

## a. Problem statement

Predict the dry bean variety from 16 numerical measurements of its shape. Compare Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and Support Vector Machine using one common held-out test set. The Streamlit application loads saved pipelines and evaluates labelled test CSV files without retraining.

## b. Dataset description

Source: [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). Citation: Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. [DOI: 10.24432/C50S4B](https://doi.org/10.24432/C50S4B). License: CC BY 4.0. The included CSV is a conversion of the original ARFF; provenance and SHA-256 hashes are in data/provenance.json.

The source contains {summary['raw_rows']:,} instances, 16 numerical input features and seven target classes. It exceeds the minimum 500 instances and 12 features. Target column: Class. There are no missing values in the checked source.

Original class counts: {counts}.

Features: {", ".join(FEATURES)}. AspectRation and roundness retain the source file's exact spellings.

Before splitting, {summary['duplicate_rows_removed']} repeated feature vectors were removed, leaving {summary['usable_rows']:,} distinct observations. Conflicting labels for identical vectors are checked and rejected. A stratified split with seed {summary['seed']} yields {summary['split_rows']['train']:,} training, {summary['split_rows']['validation']:,} validation and {summary['split_rows']['test']:,} test rows (approximately 60/20/20). All models share these rows.

StandardScaler is fitted inside the training pipeline for Logistic Regression, KNN, Gaussian Naive Bayes and SVM; the tree uses original numeric values. Validation macro F1 selects the overall model. Each model is then refitted on training plus validation data, with preprocessing refitted only on those rows. The test set remains excluded from fitting and model selection.

## c. Github Repository Link

{link}

## d. Models used

Fixed settings were chosen before evaluation; no test-set hyperparameter search was performed.

- Logistic Regression: C=1.0, lbfgs solver, max_iter=3000.
- Decision Tree: max_depth=12, min_samples_leaf=3, random_state=5692.
- KNN: n_neighbors=7, weights=distance, Euclidean distance.
- Naive Bayes: GaussianNB, var_smoothing=1e-9.
- Support Vector Machine: RBF kernel, C=3.0, gamma=scale, probability=True, random_state=5692.

### Comparison table

All entries are measured on the same {summary['split_rows']['test']:,} held-out rows in test_data.csv. Values are rounded to four decimals; full precision is retained in results/metrics.csv.

{markdown_table(["ML Model Name"]+METRICS,rows)}

Accuracy is the fraction correctly classified. Precision, recall and F1 are unweighted macro averages over the seven labels (zero for undefined class scores). AUC is macro one-vs-rest ROC AUC from predicted probabilities aligned to the model's class order. MCC is multiclass Matthews correlation coefficient. AUC measures ranking and can be high even when some class decisions are wrong.

### Observations about model performance

{markdown_table(["ML Model Name","Observation about model performance"],observations)}

### Reproduce and use the app

Use Python 3.12. Install requirements.txt, run train.py to reproduce training, and run streamlit with app.py. The saved pipelines include preprocessing; all five .joblib files are under model/. The app provides CSV upload, a model dropdown, all six metrics, a comparison table, confusion matrix, classification report and prediction download.

The uploaded CSV must contain exactly the 16 feature columns and Class, numeric finite inputs, and known uppercase labels. Column order may vary. Upload only test data (up to 5,000 rows / 5 MiB). If a class is absent, seven-class AUC is displayed as N/A; macro metrics still include all seven labels. The bundled test set contains every class.

Detailed local, BITS Lab and Community Cloud steps are in RUN_GUIDE.md. The recorded execution context is {summary['execution_context_operator_declared']}; a local run is not BITS Lab evidence.

### References

[UCI dataset and license](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). [scikit-learn: preventing leakage with pipelines](https://scikit-learn.org/stable/common_pitfalls.html). [scikit-learn: multiclass AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
"""
    (ROOT/"README.md").write_text(text,encoding="utf-8",newline="\n")
    return ROOT/"README.md"


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--github-url")
    args=parser.parse_args()
    print(build(args.github_url))
