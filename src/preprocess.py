import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean(path):
    
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'churn.csv'))
    
    # customerID drop karo
    df = df.drop('customerID', axis=1)
    
    # TotalCharges fix karo
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Target encode
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Binary columns
    binary_cols = ['gender', 'Partner', 'Dependents',
                   'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    # Multi-class → one-hot
    cat_cols = ['MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # ✅ YEH ADD KARO — NaN check aur fix
    print(f"NaN values before fix: {df.isnull().sum().sum()}")
    
    # Numeric columns mein median se fill karo
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Boolean columns mein 0 se fill karo
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].fillna(False).astype(int)
    
    # Bool columns ko int mein convert karo
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})
    
    print(f"NaN values after fix: {df.isnull().sum().sum()}")  # 0 aana chahiye
    
    return df

def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return train_test_split(
        X, y, test_size=0.2, 
        random_state=42, stratify=y  # stratify important hai!
    )