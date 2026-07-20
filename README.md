# Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning pipeline to detect fraudulent credit card transactions in a highly imbalanced dataset, comparing multiple models and tuning the best performer for real-world deployment considerations.

## Problem Statement

Banks process millions of transactions daily, and fraud makes up only a tiny fraction of them. The challenge isn't just building a model — it's building one that can reliably catch rare fraud cases without drowning genuine customers in false alarms, all while accounting for the fact that missing a fraud case is far more costly to a bank than flagging a genuine transaction for review.

## Dataset

- **Source:** [Credit Card Fraud Detection dataset (Kaggle, mlg-ulb)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns
- **Features:** `Time`, `V1`–`V28` (anonymized via PCA), `Amount`, `Class` (target: 0 = genuine, 1 = fraud)
- **After removing 1,081 duplicate rows:** 283,726 transactions, of which only **473 (0.167%)** are fraud

> Note: The dataset is not included in this repo due to size — download it from Kaggle and place it at `data/creditcard.csv`.

## Key EDA Findings

- **Extreme class imbalance:** 283,253 genuine vs. 473 fraud transactions (~578:1 ratio). A model predicting "genuine" for everything would score 99.8% accuracy while catching zero fraud — which is why accuracy is not used as an evaluation metric anywhere in this project.
- **Amount:** Heavily right-skewed with extreme outliers (median ₹22 vs. max ₹25,691). Fraud transactions showed a slightly higher median amount than genuine ones, but with significant overlap — `Amount` alone is a weak standalone predictor.
- **Time:** Engineered into an `Hour` feature (0–23). Fraud showed a relatively higher density during early morning hours (2–5 AM), likely because legitimate transaction volume drops overnight, making fraud a larger share of activity during low-traffic windows.
- **Correlation with fraud:** `V17`, `V14`, and `V12` showed the strongest correlations with `Class` (all negative, -0.25 to -0.31). No single feature separates the classes cleanly — fraud detection depends on combinations of features, motivating the use of tree-based ensemble models.

## Approach

1. **Cleaning:** Removed 1,081 duplicate rows (19 of which were fraud cases — the fraud rate slightly worsened post-cleaning).
2. **Scaling:** Used `RobustScaler` (not `StandardScaler`) on `Amount` and `Hour`, since `RobustScaler` uses median/IQR and is far less distorted by the extreme outliers present in `Amount`.
3. **Train-test split:** Stratified 80/20 split to preserve the 0.167% fraud ratio in both sets — critical given how few fraud examples exist in total.
4. **Handling imbalance:** Applied **SMOTE** to the training data only, to avoid leaking synthetic or duplicated information into the test set.
5. **Modeling:** Trained and compared Logistic Regression (baseline), Random Forest, and XGBoost.
6. **Evaluation:** Used Confusion Matrix, Precision, Recall, F1, ROC-AUC, and PR-AUC — with **PR-AUC prioritized** over ROC-AUC, since ROC-AUC can look misleadingly strong on imbalanced data (a model's false positives barely move the false positive rate when true negatives vastly outnumber true positives).
7. **Tuning:** Used `RandomizedSearchCV` (`scoring='average_precision'`) on XGBoost.

## A Data Leakage Catch Worth Mentioning

The first tuning attempt returned a cross-validated PR-AUC of **0.99999** — an unrealistic score for fraud detection. This turned out to be caused by applying SMOTE *before* the cross-validation split, which let synthetic points derived from a real fraud transaction leak into a different CV fold than the original, letting the model "validate" on data very similar to what it trained on.

**Fix:** Moved SMOTE inside an `imblearn` `Pipeline`, so it's refit independently within each CV fold using only that fold's training data. This brought the CV PR-AUC down to a realistic **0.853**.

## Model Comparison

All models trained on SMOTE-balanced training data, evaluated on the original (untouched, imbalanced) test set — fraud class (Class 1) metrics shown:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | False Positives | False Negatives |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.05 | 0.87 | 0.10 | 0.9610 | 0.6774 | 1,481 | 12 |
| Random Forest | 0.92 | 0.76 | 0.83 | 0.9487 | 0.8042 | 6 | 23 |
| XGBoost | 0.78 | 0.80 | 0.79 | 0.9711 | 0.8157 | 22 | 19 |
| **XGBoost (tuned)** | **0.79** | **0.81** | **0.80** | **0.9721** | 0.8097 | **20** | **18** |

**Why XGBoost over Random Forest despite lower precision:** Random Forest achieved a higher precision and F1 score, but missed more fraud cases (23 false negatives vs. XGBoost's 19). Since a missed fraud case is typically more costly to a bank than a false alarm, XGBoost's better recall made it the preferred model — a decision driven by business context, not just the highest single metric.

## Final Model — Tuned XGBoost

**Best hyperparameters:**
```
n_estimators: 200
max_depth: 9
learning_rate: 0.1
subsample: 1.0
colsample_bytree: 1.0
```

**Test set performance:**
- Precision: 0.79, Recall: 0.81, F1: 0.80
- ROC-AUC: 0.9721, PR-AUC: 0.8097
- 20 false positives, 18 false negatives (out of 95 actual fraud cases)

## Feature Importance

`V14` was by far the most important feature for the tuned XGBoost model (~0.65 importance), consistent with EDA findings where `V14` showed the cleanest visual separation between fraud and genuine transactions of any feature. Other notable features: `V4`, `V10`, `V8`, `V12`.

Interestingly, `V17` — the feature with the *strongest correlation* to fraud in EDA — did not rank in the top 10 by model importance, likely because `V14` captures much of the same separating signal, making `V17` less necessary for the trees to split on repeatedly.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML:** Scikit-learn, XGBoost
- **Imbalanced Learning:** imbalanced-learn (SMOTE)

## How to Run

```bash
git clone <repo-url>
cd Credit-Card-Fraud-Detection
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in `data/`, then open `notebooks/fraud_detection.ipynb`.

## Folder Structure

```
Credit-Card-Fraud-Detection/
├── data/
│   └── creditcard.csv          # not included — download from Kaggle
├── notebooks/
│   └── fraud_detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Takeaways

- Accuracy is meaningless on datasets this imbalanced — PR-AUC and Recall are the metrics that actually matter.
- Resampling techniques like SMOTE must be applied carefully with respect to train/test and cross-validation boundaries, or they silently inflate performance metrics.
- The "best" model isn't always the one with the highest single metric — it depends on the real-world cost of false positives vs. false negatives for the specific business problem.