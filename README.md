# Machine Learning Assignment 2 - Dry Bean Classification

DINESH BEHERA | Student | BITS ID: 2026AJ05692 | AIML ZG565

## a. Problem statement

Predict the dry bean variety from 16 numerical measurements of its shape. Compare Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and Support Vector Machine using one common held-out test set. The Streamlit application loads saved pipelines and evaluates labelled test CSV files without retraining.

## b. Dataset description

Source: [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). Citation: Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. [DOI: 10.24432/C50S4B](https://doi.org/10.24432/C50S4B). License: CC BY 4.0. The included CSV is a conversion of the original ARFF; provenance and SHA-256 hashes are in data/provenance.json.

The source contains 13,611 instances, 16 numerical input features and seven target classes. It exceeds the minimum 500 instances and 12 features. Target column: Class. There are no missing values in the checked source.

Original class counts: DERMASON: 3546, SIRA: 2636, SEKER: 2027, HOROZ: 1928, CALI: 1630, BARBUNYA: 1322, BOMBAY: 522.

Features: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4. AspectRation and roundness retain the source file's exact spellings.

Before splitting, 68 repeated feature vectors were removed, leaving 13,543 distinct observations. Conflicting labels for identical vectors are checked and rejected. A stratified split with seed 5692 yields 8,125 training, 2,709 validation and 2,709 test rows (approximately 60/20/20). All models share these rows.

StandardScaler is fitted inside the training pipeline for Logistic Regression, KNN, Gaussian Naive Bayes and SVM; the tree uses original numeric values. Validation macro F1 selects the overall model. Each model is then refitted on training plus validation data, with preprocessing refitted only on those rows. The test set remains excluded from fitting and model selection.

## c. Github Repository Link

[GitHub repository](https://github.com/DinuDante/dry-bean-classifier)

[Live Streamlit app](https://dinudante-dry-bean-classifier.streamlit.app)

## d. Models used

Fixed settings were chosen before evaluation; no test-set hyperparameter search was performed.

- Logistic Regression: C=1.0, lbfgs solver, max_iter=3000.
- Decision Tree: max_depth=12, min_samples_leaf=3, random_state=5692.
- KNN: n_neighbors=7, weights=distance, Euclidean distance.
- Naive Bayes: GaussianNB, var_smoothing=1e-9.
- Support Vector Machine: RBF kernel, C=3.0, gamma=scale, probability=True, random_state=5692.

### Comparison table

All entries are measured on the same 2,709 held-out rows in test_data.csv. Values are rounded to four decimals; full precision is retained in results/metrics.csv.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9243 | 0.9949 | 0.9384 | 0.9352 | 0.9365 | 0.9085 |
| Decision Tree | 0.9014 | 0.9642 | 0.9145 | 0.9131 | 0.9138 | 0.8808 |
| KNN | 0.9291 | 0.9877 | 0.9429 | 0.9384 | 0.9405 | 0.9142 |
| Naive Bayes | 0.8989 | 0.9909 | 0.9099 | 0.9105 | 0.9098 | 0.8780 |
| Support Vector Machine | 0.9328 | 0.9959 | 0.9461 | 0.9418 | 0.9439 | 0.9187 |

Accuracy is the fraction correctly classified. Precision, recall and F1 are unweighted macro averages over the seven labels (zero for undefined class scores). AUC is macro one-vs-rest ROC AUC from predicted probabilities aligned to the model's class order. MCC is multiclass Matthews correlation coefficient. AUC measures ranking and can be high even when some class decisions are wrong.

### Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Lowest class recall: SIRA (0.8653). A regularized linear boundary provides a useful baseline after standardization; remaining errors are consistent with overlapping class shapes. |
| Decision Tree | Lowest class recall: SIRA (0.8425). The depth-12 tree captures nonlinear feature thresholds without scaling. Its axis-aligned splits can miss smoother boundaries; depth and leaf limits restrict overfitting. |
| KNN | Lowest class recall: SIRA (0.8672). Distance-weighted voting with seven neighbors benefits from scaling because raw area and shape ratios have very different magnitudes. |
| Naive Bayes | Lowest class recall: BARBUNYA (0.8075). Gaussian conditional independence is restrictive here because area, convex area and axis lengths are related. This is a plausible explanation, not a causal test. |
| Support Vector Machine | Lowest class recall: SIRA (0.8653). An RBF kernel models nonlinear boundaries in standardized feature space. Probabilities are fitted internally on development data, without the held-out test. |
| Overall winner | Support Vector Machine, selected by validation macro F1 (0.9395) before test evaluation. Its test macro F1 is 0.9439. The highest test macro F1 in this run belongs to Support Vector Machine (0.9439). These are results of one fixed split, not a statistical superiority claim. |

### Reproduce and use the app

Use Python 3.12. Install requirements.txt, run train.py to reproduce training, and run streamlit with app.py. The saved pipelines include preprocessing; all five .joblib files are under model/. The app provides CSV upload, a model dropdown, all six metrics, a comparison table, confusion matrix, classification report and prediction download.

The uploaded CSV must contain exactly the 16 feature columns and Class, numeric finite inputs, and known uppercase labels. Column order may vary. Upload only test data (up to 5,000 rows / 5 MiB). If a class is absent, seven-class AUC is displayed as N/A; macro metrics still include all seven labels. The bundled test set contains every class.

Detailed local, BITS Lab and Community Cloud steps are in RUN_GUIDE.md. The recorded execution context is bits-lab; the genuine lab screenshot and checked execution logs accompany the final report.

### References

[UCI dataset and license](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). [scikit-learn: preventing leakage with pipelines](https://scikit-learn.org/stable/common_pitfalls.html). [scikit-learn: multiclass AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
