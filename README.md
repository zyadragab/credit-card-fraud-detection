# 💳 Credit Card Fraud Detection — End-to-End ML Project
 
A complete machine learning pipeline to detect fraudulent credit card transactions, tackling one of the most challenging real-world problems in fintech: **extreme class imbalance**.
 
🚀 **[Live Demo — Try the App](https://credit-card-fraud-detection-hopygpheevrjofre8rsngh.streamlit.app/)**
 
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
→ SMOTE → Modeling → Cross Validation → Overfitting Check
→ Evaluation → SHAP Analysis → Threshold Tuning → Streamlit Deployment
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
 
| Model | AUC-ROC (CV) |
|---|---|
| Logistic Regression | ~0.96 |
| Random Forest | ~0.97 |
| XGBoost | ~0.95 |
 
---
 
## 🔎 Overfitting Check
 
| Model | Train AUC | Test AUC | Gap | Status |
|---|---|---|---|---|
| Logistic Regression | 0.9897 | 0.9619 | 0.028 | ✅ OK |
| Random Forest | 1.0000 | 0.9656 | 0.034 | ✅ OK |
| XGBoost | 1.0000 | 0.9495 | 0.051 | ⚠️ Slight Overfit |
 
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
 
- **Highest AUC-PR: 0.8038**
- **Precision: 0.91** — when it flags fraud, it's right 91% of the time
- **Only 7 false alarms** vs Logistic Regression's 1,482
> Logistic Regression had higher Recall (87%) but generated **1,482 false alarms**. In production, every false alarm means a real customer getting their card blocked. Precision matters.
 
---
 
## 🧠 SHAP Analysis — Explainability
 
Used **SHAP (SHapley Additive Explanations)** to explain why the model flags a transaction as fraud.
 
- `V14` — strongest signal (negative values strongly indicate fraud)
- `V12`, `V4`, `V3`, `V10` — also highly influential
---
 
## 🎯 Threshold Tuning
 
| Threshold | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.50 (default) | 0.91 | 0.76 | — |
| **0.40 (optimal)** | **0.87** | **0.79** | **0.83** |
 
---
 
## 🖥️ Streamlit App
 
🔗 **[Try the live demo](https://credit-card-fraud-detection-hopygpheevrjofre8rsngh.streamlit.app/)**
 
---
 
## 📁 Project Structure
 
```
fraud-detection/
│
├── fraud_detection.ipynb    # Main notebook — full pipeline
├── app.py                   # Streamlit web app
├── fraud_model.pkl          # Trained Random Forest model
├── scaler.pkl               # Fitted StandardScaler
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
streamlit
joblib
```
 
---
 
## 🚀 How to Run Locally
 
```bash
pip install -r requirements.txt
streamlit run app.py
```
 
---
 
## 👤 Author
 
**Zyad Ragab** — AI Student & Aspiring ML Engineer
📎 [LinkedIn](https://www.linkedin.com/in/zyad-ragab-3a4086337/) | 💻 [GitHub](https://github.com/zyadragab)
