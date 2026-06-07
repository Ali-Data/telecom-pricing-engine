"""
Multi-Class Churn & Downgrade Prediction Engine
-----------------------------------------------
This script trains a calibrated XGBoost classifier to predict 3 customer states:
Class 0: Accept (Renew with same or higher ARPU)
Class 1: Downgrade (Renew with lower ARPU)
Class 2: Churn (Fail to renew within 15-day grace period)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path):
    """Loads dbt output, drops censored instances (grace period NULLs), and prepares X, y."""
    print(f"[*] Loading data from {file_path}...")
    
    # In a real pipeline, this could be a SQL Alchemy connection to your Data Warehouse
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("ERROR: Data file not found. Ensure dbt has generated the CSV export.")
        return None, None, None
        
    # Drop rows where customer action is NULL (Users currently in the 15-day grace period)
    df_clean = df.dropna(subset=['customer_action']).copy()
    df_clean['customer_action'] = df_clean['customer_action'].astype(int)
    
    # Drop identifying and text columns, plus actual_paid_price (machine uses catalog_price and discount)
    cols_to_drop = ['username', 'purchaseid', 'purchasedate', 'province', 'salesagent', 'actual_paid_price']
    
    # Dynamically drop columns only if they exist in the dataframe (case-insensitive handling)
    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    
    X = df_clean.drop(columns=cols_to_drop + ['customer_action'])
    y = df_clean['customer_action']
    
    print(f"[*] Dataset shape after preprocessing: {X.shape}")
    print(f"[*] Class Distribution:\n{y.value_counts(normalize=True).round(3) * 100}")
    
    return X, y, df_clean

def train_and_tune_model(X_train, y_train, train_weights):
    """Trains an XGBoost classifier with randomized hyperparameter tuning."""
    print("\n[*] Initializing Hyperparameter Tuning (RandomizedSearchCV)...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_base = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=3,                
        eval_metric='mlogloss',     
        random_state=42,
        n_jobs=-1
    )
    
    # Using RandomizedSearch to find the best configuration
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=15,                  
        scoring='neg_log_loss',     
        cv=3,                       
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("[*] Training the base AI brain. This might take a few minutes...")
    search.fit(X_train, y_train, sample_weight=train_weights)
    
    print(f"\n[*] Best Parameters Found:\n{search.best_params_}")
    return search.best_estimator_

def calibrate_model(best_model, X_train, y_train, train_weights):
    """Wraps the best model in an Isotonic Regression layer for true probability extraction."""
    print("\n[*] Applying Probability Calibration (Isotonic Regression)...")
    
    calibrated_xgb = CalibratedClassifierCV(
        estimator=best_model, 
        method='isotonic', 
        cv=3,
        n_jobs=-1
    )
    calibrated_xgb.fit(X_train, y_train, sample_weight=train_weights)
    
    return calibrated_xgb

def main():
    # Define file paths (assuming script is run from project root or ml_pipeline folder)
    # You may need to adjust this path based on where you save your dbt CSV output
    data_path = "marts_telecom_features.csv" 
    model_save_path = "ml_pipeline/calibrated_xgb_model.pkl"
    feature_names_path = "ml_pipeline/feature_names.pkl"
    
    # 1. Load Data
    X, y, df_raw = load_and_preprocess_data(data_path)
    if X is None:
        return
        
    # 2. Split and Weight
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n[*] Computing sample weights for imbalanced classes...")
    train_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 3. Train & Tune
    best_xgb_model = train_and_tune_model(X_train, y_train, train_weights)
    
    # 4. Calibrate Probabilities
    calibrated_model = calibrate_model(best_xgb_model, X_train, y_train, train_weights)
    
    # 5. Evaluate on Unseen Test Data
    print("\n[*] Evaluating Final Calibrated Model on Test Data...")
    y_pred = calibrated_model.predict(X_test)
    target_names = ['Class 0 (Accept)', 'Class 1 (Downgrade)', 'Class 2 (Churn)']
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 6. Save Model and Feature Names for the Simulator
    print("\n[*] Saving model artifacts for the Dynamic Simulator...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(calibrated_model, model_save_path)
    joblib.dump(list(X.columns), feature_names_path)
    print(f"SUCCESS: Model saved to {model_save_path}")

if __name__ == "__main__":
    main()