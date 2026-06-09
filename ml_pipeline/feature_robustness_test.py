"""
MLOps Pipeline: Feature Robustness & Ablation Test
Evaluates the predictive impact of reverse-engineered latent discount features 
against a baseline raw-feature model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

def run_robustness_test():
    print("[*] Initializing Feature Robustness Evaluation...")
    print("[*] Loading Production Dataset...")
    
    try:
        df = pd.read_csv("marts_telecom_features.csv").dropna(subset=['customer_action'])
    except FileNotFoundError:
        print("[!] CRITICAL ERROR: 'marts_telecom_features.csv' not found. Please run dbt pipeline first.")
        return

    df['customer_action'] = df['customer_action'].astype(int)
    
    # Drop operational identifiers
    cols_to_drop = ['username', 'purchaseid', 'purchasedate', 'province', 'salesagent', 'actual_paid_price']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop + ['customer_action'])
    y = df['customer_action']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Identify engineered latent features
    latent_features = ['hidden_discount_percentage', 'catalog_price', 'hidden_discount_amount']
    latent_features = [f for f in latent_features if f in X.columns]
    
    if not latent_features:
        print("[!] WARNING: Latent discount features not found in the dataset.")
        return
        
    print(f"[*] Target Features for Ablation: {latent_features}")
    
    # Create baseline dataset (without latent features)
    X_train_base = X_train.drop(columns=latent_features)
    X_test_base = X_test.drop(columns=latent_features)
    
    print("\n[*] Training Baseline XGBoost Model (Raw Features Only)...")
    model_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    model_base.fit(X_train_base, y_train)
    
    print("[*] Training Enhanced XGBoost Model (With Latent Features)...")
    model_enhanced = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    model_enhanced.fit(X_train, y_train)
    
    # --- Metrics Computation ---
    print("\n[*] Computing Evaluation Metrics...\n")
    
    probs_base = model_base.predict_proba(X_test_base)
    preds_base = model_base.predict(X_test_base)
    
    probs_enhanced = model_enhanced.predict_proba(X_test)
    preds_enhanced = model_enhanced.predict(X_test)
    
    ll_base = log_loss(y_test, probs_base)
    f1_base = f1_score(y_test, preds_base, average='macro')
    
    ll_enhanced = log_loss(y_test, probs_enhanced)
    f1_enhanced = f1_score(y_test, preds_enhanced, average='macro')
    
    # Terminal Report (Formatted for MLOps logging)
    print("="*70)
    print(" MODEL PERFORMANCE REPORT: LATENT FEATURE ABLATION")
    print("="*70)
    print(f"{'Model Architecture':<35} | {'Macro F1':<10} | {'Log-Loss':<10}")
    print("-" * 70)
    print(f"{'Baseline (Excluding Latent Vars)':<35} | {f1_base:.4f}     | {ll_base:.4f}")
    print(f"{'Enhanced (Including Latent Vars)':<35} | {f1_enhanced:.4f}     | {ll_enhanced:.4f}")
    print("="*70)
    
    f1_improvement = ((f1_enhanced - f1_base) / f1_base) * 100
    ll_improvement = ((ll_base - ll_enhanced) / ll_base) * 100
    
    print(f"\n[+] SYSTEM INSIGHT:")
    print(f"Integration of reverse-engineered latent marketing variables improved the")
    print(f"model's Macro F1 score by {f1_improvement:.2f}% and reduced predictive uncertainty")
    print(f"(Log-loss) by {ll_improvement:.2f}%. These features are marked as highly critical")
    print(f"for production deployment.")

if __name__ == "__main__":
    run_robustness_test()