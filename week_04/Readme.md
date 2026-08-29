# Machine Learning Lab

## Student Details

**Name:** Akshay Kumar M  
**SRN:** PES1UG24CS045  

---

## Objective

This lab focuses on implementing and comparing different hyperparameter tuning techniques for machine learning classification models. The experiments include manual grid search, `GridSearchCV`, and `RandomizedSearchCV`.

The lab also explores the effect of feature scaling and feature selection on model performance and combines multiple classifiers using voting classifiers.

---

## Learning Objectives

- Understand hyperparameter tuning using exhaustive and randomized search.
- Implement manual grid search using cross-validation.
- Compare manual grid search with `GridSearchCV`.
- Use `RandomizedSearchCV` for efficient hyperparameter optimization.
- Analyze the effect of `StandardScaler` and `MinMaxScaler`.
- Perform feature selection using `SelectKBest`.
- Train and evaluate Decision Tree and k-Nearest Neighbors classifiers.
- Create and evaluate voting classifiers.
- Analyze model performance using ROC curves and confusion matrices.

---

## Machine Learning Models Used

The following classification algorithms are used:

- Decision Tree Classifier
- k-Nearest Neighbors (kNN)
- Voting Classifier

---

## Hyperparameter Tuning Methods

### 1. Manual Grid Search

A custom implementation is used to generate different combinations of hyperparameters and evaluate them using 5-fold stratified cross-validation.

### 2. GridSearchCV

Scikit-learn's `GridSearchCV` is used to automatically perform an exhaustive search over specified hyperparameter combinations.

### 3. RandomizedSearchCV

`RandomizedSearchCV` is used to randomly sample hyperparameter combinations and identify suitable configurations efficiently.

---

## Datasets Used

The lab uses the following datasets:

1. **Wine Quality Dataset**
   - Predicts wine quality based on chemical properties.

2. **HR Attrition Dataset**
   - Predicts employee turnover.

3. **Banknote Authentication Dataset**
   - Detects whether a banknote is authentic or counterfeit.

4. **QSAR Biodegradation Dataset**
   - Predicts the biodegradability of chemical compounds.

---

## Techniques Used

- Train-Test Split
- Stratified K-Fold Cross Validation
- Feature Scaling
- Feature Selection
- Hyperparameter Tuning
- Pipeline Construction
- Model Evaluation
- Ensemble Learning

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- ROC Curve
- Confusion Matrix
- Classification Report

---

## Libraries Used

- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

---

## Project Structure

```text
ML-Lab/
│
├── ML_Lab.ipynb
└── README.md