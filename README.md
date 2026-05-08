# 📡 Customer Churn Predictor with SHAP Explainability

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat-square&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Predict which telecom customers are likely to churn — and explain *why* using SHAP.**  
> Built with XGBoost, SHAP, and deployed as an interactive Streamlit web app.

🔗 **Live Demo:** [customer-churn-predictor-umyhgzcdydxzapps2yhz6uh.streamlit.app](https://customer-churn-predictor-umyhgzcdydxzapps2yhz6uh.streamlit.app)

---

## 🎯 What This Project Does

Most churn models just say *"this customer will leave"* — this one tells you **why**.

Enter a customer's details → get churn probability + a SHAP waterfall chart showing which factors drove the prediction. This is the kind of **actionable, explainable AI** that real retention teams actually use.

---

## 🖼️ Demo

| Prediction | SHAP Explanation |
|---|---|
| ✅ Low Risk / ⚠️ High Risk with probability score | Waterfall chart showing top contributing features |

---

## 🚀 Features

- **Churn probability prediction** — real-time, per customer
- **SHAP explainability** — waterfall chart shows *why* the model predicted what it did
- **Business recommendations** — actionable retention strategies based on risk level
- **Interactive UI** — built with Streamlit, no technical knowledge needed to use
- **Class imbalance handling** — SMOTE oversampling for better recall on minority class
- **ROC-AUC: 0.83** — strong performance on held-out test set

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📁 Project Structure

```
customer-churn-predictor/
├── data/
│   └── churn.csv              # Telco Customer Churn dataset (Kaggle)
├── models/
│   └── xgb_model.pkl          # Trained XGBoost model
├── notebooks/
│   └── 01_eda_and_train.ipynb # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train.py               # Model training + evaluation
│   └── explain.py             # SHAP explainability logic
├── app.py                     # Streamlit web application
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Dataset

**Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Features | 20 input features |
| Target | Churn (Yes / No) |
| Churn Rate | ~26.5% (class imbalance handled via SMOTE) |

---

## 🧠 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **0.8351** |
| Accuracy | 78% |
| Precision (churn) | 59% |
| Recall (churn) | 63% |

> Recall optimized intentionally — catching churners matters more than false alarms in retention strategy.

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/disha1647/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train.py

# 5. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔍 How SHAP Works Here

SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a specific prediction.

- 🔴 **Red bars** → feature *increases* churn risk
- 🟢 **Green bars** → feature *decreases* churn risk

**Example insight:** A customer on a month-to-month contract with high monthly charges and only 3 months tenure = very high churn risk. SHAP shows exactly which of these factors matters most — so retention teams can act on it.

---

## 💡 Business Value

This tool helps telecom retention teams:
- Identify at-risk customers **before** they leave
- Understand **why** they're at risk
- Take **targeted action** — discount offer, contract upgrade, tech support

---

## 👩‍💻 Built By

**Disha** — AI/ML Engineer  
B.Tech in AI/ML | M.Tech (pursuing)  
Specializing in: LLMs, RAG pipelines, agentic AI, and production ML systems

📬 [GitHub](https://github.com/disha1647) · [LinkedIn](#) · [Upwork](#)

---

## 📄 License

MIT License — free to use, modify, and distribute.
