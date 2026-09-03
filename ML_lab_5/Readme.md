# Lab 5: Linear vs. Logistic Regression and Threshold Tuning

**Course:** UE24CS352A – Machine Learning
**Name:** Aditya TJ
**SRN:** PES1UG24CS031

## Objective
Implement and compare linear regression and logistic regression for binary classification. Construct a robust machine learning pipeline and evaluate the impact of classification thresholds on model performance.

## Files in this Submission
| File | Description |
|---|---|
| `Lab5_Linear_vs_Logistic_Regression.ipynb` | Fully executed Jupyter/Colab notebook containing all code, outputs, and plots for Steps 1–9. |
| `Lab5_Report.pdf` | Formal lab report with methodology, results, analysis, and answers to the conceptual questions. |
| `README.md` | This file. |

## How to Run
1. Open `Lab5_Linear_vs_Logistic_Regression.ipynb` in Google Colab.
2. Select **Runtime → Run all**.
3. The notebook downloads the Wine Quality (Red) dataset directly from the UCI repository, so an active internet connection is required.
4. All cells run top to bottom with no manual edits needed.

## Datasets Used
- **1D Toy Dataset:** Synthetically generated ("Hours Studied" vs. "Pass/Fail") with an injected extreme outlier at X = 50, used to visualize decision boundaries.
- **Wine Quality (Red) Dataset:** UCI Machine Learning Repository — [`winequality-red.csv`](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv). Target is binarized as `good_quality = 1` if `quality > 5`, else `0`.

## Pipeline Summary
```
StandardScaler → SelectKBest(f_classif, k=8) → LogisticRegression(random_state=42)
```
- Train/test split: 70/30, stratified on target, `random_state=42`.
- Probabilities extracted via `predict_proba()`.
- Manual thresholds applied: **0.3, 0.5, 0.7**.

## Key Results
| Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0.3 | 0.700 | 0.657 | 0.922 | 0.767 |
| 0.5 | 0.731 | 0.752 | 0.743 | 0.748 |
| 0.7 | 0.694 | 0.853 | 0.518 | 0.644 |

**Takeaway:** As the threshold rises, Precision increases while Recall decreases — the classic Precision/Recall trade-off. F1-Score peaks at the default threshold of 0.5 in this run.

## Dependencies
```
pandas
numpy
matplotlib
scikit-learn
```

## Notes
- Random seeds (`42`) are fixed throughout for reproducibility.
- See `Lab5_Report.pdf` for the full write-up, all plots, confusion matrices, and answers to the five conceptual questions (Precision/Recall trade-off, contextual threshold selection, MSE vs. Log-Loss, and gradient descent convexity).