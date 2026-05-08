import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import load_and_clean, split_data
from src.explain import get_shap_explainer, plot_waterfall, get_top_reasons

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ─── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_everything():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'xgb_model.pkl')
    model = joblib.load(model_path)
    
    df = load_and_clean(os.path.join(BASE_DIR, 'data', 'churn.csv'))
    X_train, X_test, y_train, y_test = split_data(df)
    explainer = get_shap_explainer(model)
    
    return model, explainer, X_test.columns.tolist()

model, explainer, feature_cols = load_everything()

# ─── HEADER ────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.markdown("**XGBoost + SHAP Explainability** | Built by Disha")
st.divider()

# ─── INPUT FORM ────────────────────────────────────────────
st.subheader("🧾 Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    senior_citizen = st.checkbox("Senior Citizen")
    partner = st.checkbox("Has Partner")
    dependents = st.checkbox("Has Dependents")
    paperless_billing = st.checkbox("Paperless Billing", value=True)
    phone_service = st.checkbox("Phone Service", value=True)

st.divider()

# ─── PREDICT BUTTON ────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):

    # ── Build input dict ──────────────────────────────────
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': int(senior_citizen),
        'Partner': int(partner),
        'Dependents': int(dependents),
        'PhoneService': int(phone_service),
        'PaperlessBilling': int(paperless_billing),
        'gender': 0,  # default
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': int(internet_service == 'Fiber optic'),
        'InternetService_No': int(internet_service == 'No'),
        'OnlineSecurity_No internet service': int(online_security == 'No internet service'),
        'OnlineSecurity_Yes': int(online_security == 'Yes'),
        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': int(tech_support == 'No internet service'),
        'TechSupport_Yes': int(tech_support == 'Yes'),
        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': int(contract == 'One year'),
        'Contract_Two year': int(contract == 'Two year'),
        'PaymentMethod_Credit card (automatic)': int(payment_method == 'Credit card (automatic)'),
        'PaymentMethod_Electronic check': int(payment_method == 'Electronic check'),
        'PaymentMethod_Mailed check': int(payment_method == 'Mailed check'),
    }

    # ── Align columns with training data ──────────────────
    input_df = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    # ── Prediction ────────────────────────────────────────
    prob = model.predict_proba(input_df)[0][1]
    prediction = prob > 0.5

    # ── Result display ────────────────────────────────────
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        if prediction:
            st.error(f"⚠️ High Churn Risk")
        else:
            st.success(f"✅ Low Churn Risk")
    with col_res2:
        st.metric("Churn Probability", f"{prob:.1%}")
    with col_res3:
        st.metric("Retention Probability", f"{1-prob:.1%}")

    st.divider()

    # ── SHAP Explanation ──────────────────────────────────
    st.subheader("🔍 Why this prediction?")

    # Top reasons text
    reasons = get_top_reasons(explainer, input_df, n=5)
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Top factors:**")
        for feat, val in reasons:
            if val > 0:
                st.markdown(f"🔴 `{feat}` — increases churn risk")
            else:
                st.markdown(f"🟢 `{feat}` — decreases churn risk")

    with col_r2:
        # Waterfall chart
        try:
            fig = plot_waterfall(explainer, input_df)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info("SHAP chart load ho raha hai...")

    st.divider()

    # ── Business Recommendation ───────────────────────────
    st.subheader("💡 Recommended Action")
    if prob > 0.7:
        st.warning("""
        **High Risk Customer — Immediate action needed!**
        - Offer 2-year contract discount
        - Assign dedicated support agent
        - Provide loyalty rewards
        """)
    elif prob > 0.4:
        st.info("""
        **Medium Risk — Monitor closely**
        - Send retention email
        - Offer upgrade or bundle deal
        """)
    else:
        st.success("""
        **Low Risk — Customer is happy!**
        - Continue regular engagement
        - Upsell opportunity
        """)

# ─── FOOTER ────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px'>Built by Disha | XGBoost + SHAP | Portfolio Project</p>",
    unsafe_allow_html=True
)