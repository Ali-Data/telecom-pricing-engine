"""
MLOps Pipeline: Automated Model Retraining Orchestrator
Fetches the latest feature store data from dbt, retrains the Prescriptive 
XGBoost Engine, and serializes the model artifact for production UI (Streamlit).
"""

import pandas as pd
import xgboost as xgb
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def run_training_pipeline():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [*] Initiating Automated Retraining Pipeline...")
    
    # 1. Load Latest Data
    data_path = "marts_telecom_features.csv"
    try:
        print("[*] Fetching latest features from dbt output...")
        df = pd.read_csv(data_path).dropna(subset=['customer_action'])
    except FileNotFoundError:
        print(f"[!] CRITICAL ERROR: Feature store '{data_path}' not found.")
        return

    # 2. Data Preparation
    df['customer_action'] = df['customer_action'].astype(int)
    cols_to_drop = ['username', 'purchaseid', 'purchasedate', 'province', 'salesagent', 'actual_paid_price']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop + ['customer_action'])
    y = df['customer_action']
    
    # 3. Train Production Model
    print("[*] Retraining Prescriptive XGBoost Engine with new data...")
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=3, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    
    # 4. Save Model Artifact
    artifact_dir = "models"
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
        
    model_path = os.path.join(artifact_dir, "production_xgboost_model.json")
    model.save_model(model_path)
    
    print(f"[+] SUCCESS: New model artifact successfully saved to '{model_path}'")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [*] Pipeline execution completed. Streamlit UI will now use the updated model.")

if __name__ == "__main__":
    run_training_pipeline()