import xgboost as xgb
import joblib
import os
import sys
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

sys.path.insert(0, r"C:\Users\Disha Sagar\OneDrive\Desktop\Customer-churn")

from src.preprocess import load_and_clean, split_data
from src.explain import get_shap_explainer, plot_summary, get_top_reasons

def train_model():
    # Data load karo
    df = load_and_clean("data\\churn.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # SMOTE — class imbalance handle karo
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # Model define karo
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    
    # Train karo
    model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # Evaluate karo
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Model save karo
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'xgb_model.pkl'))
    print("✅ Model saved!")
    
    return model, X_test

if __name__ == '__main__':
    # Train karo
    model, X_test = train_model()
    
    # SHAP — sab kuch if block ke ANDAR hai ab
    explainer = get_shap_explainer(model)
    
    reasons = get_top_reasons(explainer, X_test.iloc[[0]])
    print("\n🔍 Top churn reasons for customer 1:")
    for feat, val in reasons:
        direction = "⬆️ increases" if val > 0 else "⬇️ decreases"
        print(f"  {feat}: {direction} churn risk ({val:.3f})")
    
    fig = plot_summary(explainer, X_test)
    fig.savefig('shap_summary.png', bbox_inches='tight', dpi=150)
    print("\n✅ SHAP summary saved as shap_summary.png")