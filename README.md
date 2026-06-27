# 💳 Credit Card Fraud Detection — End-to-End ML Project

A complete machine learning pipeline to detect fraudulent credit card transactions, tackling one of the most challenging real-world problems in fintech: **extreme class imbalance**.

---

## 📌 Problem Statement

Credit card fraud costs the global financial industry billions of dollars annually. The core challenge is not just building a model — it's building one that works on **severely imbalanced data**, where fraud cases represent less than **0.2% of all transactions**.

A naive model that predicts "Legit" for everything achieves **99.8% accuracy** — yet catches **zero fraud**. This project addresses that challenge head-on.

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.17%)
- **Features:** 30 (V1–V28 are PCA-transformed for confidentiality, plus `Time` and `Amount`)
- **Target:** `Class` — 0 (Legit) / 1 (Fraud)

---

## ⚙️ Project Pipeline

```
Data Loading → EDA → Preprocessing → Data Splitting → Feature Scaling
→ SMOTE → Modeling → Cross Validation → Evaluation → SHAP Analysis → Threshold Tuning
```

---

## 🔍 Exploratory Data Analysis (EDA)

- Confirmed **extreme class imbalance**: 99.83% Legit vs 0.17% Fraud
- Analyzed transaction `Amount` distribution across both classes
- Identified top features correlated with fraud using a correlation heatmap
- **Key insight:** A dummy model predicting "Legit" for everything achieves 99.8% accuracy — proving that accuracy is a meaningless metric on imbalanced data

---

## 🛠️ Preprocessing & Feature Engineering

### Why Scale Only `Time` and `Amount`?
The V1–V28 features are already PCA-transformed and standardized. Only `Time` and `Amount` needed scaling using `StandardScaler`.

> ⚠️ Scaler was fit **only on the training set** and applied to the test set — preventing data leakage.

### Handling Class Imbalance — SMOTE
**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic fraud samples in the training set to balance the classes.

**Why SMOTE instead of just class weights?**
SMOTE creates new data points by interpolating between existing minority samples, giving the model richer patterns to learn from — not just repeated copies.

> ⚠️ SMOTE was applied **inside the pipeline** using `ImbPipeline` to ensure it only runs on training folds during Cross Validation — preventing data leakage into validation folds.

---

## 🤖 Models

Three models were trained and compared:

| Model | Reason for Selection |
|---|---|
| **Logistic Regression** | Simple baseline — fast and interpretable |
| **Random Forest** | Strong ensemble model, resistant to overfitting |
| **XGBoost** | Sequential boosting — each tree corrects the previous one's errors using gradient optimization |

All models were wrapped in `ImbPipeline` with SMOTE to ensure correct behavior during Cross Validation.

---

## ✅ Cross Validation

Used **Stratified K-Fold (5 folds)** to evaluate models reliably.

**Why Stratified?**
With only 0.17% fraud cases, random splitting could result in folds with zero fraud samples. Stratified KFold ensures each fold maintains the original class ratio.

**Why Cross Validation over a single train/test split?**
A single split can be lucky or unlucky depending on which samples end up in the test set. CV averages over 5 different splits for a more reliable estimate.

| Model | AUC-ROC (CV) |
|---|---|
| Logistic Regression | ~0.96 |
| Random Forest | ~0.97 |
| XGBoost | ~0.95 |

---

## 📊 Evaluation Results

> Accuracy was ignored. Focus metrics: **Precision, Recall, F1-Score, AUC-ROC, AUC-PR**

| Model | Fraud Precision | Fraud Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| Logistic Regression | 0.05 | 0.87 | 0.9619 | 0.6769 |
| Random Forest | **0.91** | 0.76 | **0.9656** | **0.8038** |
| XGBoost | 0.54 | 0.81 | 0.9495 | 0.7941 |

---

## 🏆 Best Model — Random Forest

**Why Random Forest?**

The key metric for fraud detection is **AUC-PR (Area Under Precision-Recall Curve)** — not AUC-ROC. AUC-PR directly measures the tradeoff between catching fraud (Recall) and avoiding false alarms (Precision), which matters most in imbalanced problems.

- **Highest AUC-PR: 0.8038** — best overall balance between Precision and Recall
- **Precision: 0.91** — when it flags fraud, it's right 91% of the time
- **Practical value:** False alarms (flagging a legit transaction as fraud) cost money and damage customer trust — high Precision matters

> Logistic Regression had higher Recall (0.87) but generated **1,482 false alarms** vs Random Forest's **7** — completely impractical in production.

---

## 🧠 SHAP Analysis — Explainability

Used **SHAP (SHapley Additive Explanations)** with `TreeExplainer` to understand *why* the model flags a transaction as fraud.

**Most important features for fraud detection:**
- `V14` — strongest signal (negative values strongly indicate fraud)
- `V12`, `V4`, `V3`, `V10` — also highly influential

**Why SHAP matters:**
Banks and financial institutions can't deploy a "black box" — they need to explain every fraud decision to regulators and customers. SHAP makes that possible.

---

## 🎯 Threshold Tuning

By default, models use a **0.5 probability threshold** to classify fraud. We tuned this threshold to optimize for our business goal.

| Threshold | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.50 (default) | 0.91 | 0.76 | — |
| **0.40 (optimal)** | **0.87** | **0.79** | **0.83** |

**Why lower the threshold?**
Lowering from 0.5 to 0.4 catches **3 more fraud cases per 100** at the cost of a small drop in Precision — a worthwhile tradeoff in fraud detection where missing fraud is more costly than a false alarm.

---

## 📁 Project Structure

```
fraud-detection/
│
├── fraud_detection.ipynb    # Main notebook — full pipeline
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## 🔧 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
```

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Upload `creditcard.csv.zip` to Google Colab
3. Open `fraud_detection.ipynb` and run all cells

---

## 👤 Author

**Zyad** — AI Student & Aspiring ML Engineer  
📎 [LinkedIn](#) | 💻 [GitHub](#)
