import shap
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def plot_summary(explainer, X_test):
    """Global — kaunsa feature sabse important hai overall"""
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    return fig

def plot_waterfall(explainer, customer_row):
    """Single customer — kyun yeh churn karega"""
    shap_values = explainer(customer_row)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    return fig

def get_top_reasons(explainer, customer_row, n=5):
    """Top N reasons human-readable"""
    shap_vals = explainer.shap_values(customer_row)[0]
    feature_names = customer_row.columns.tolist()
    reasons = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:n]
    return reasons