"""
MLOps Pipeline: Model Sanity Check & Elasticity Validation
Simulates a high-risk (angry) customer to prove the prescriptive engine 
dynamically allocates aggressive discounts when mathematically justified.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def run_sanity_check():
    print("[*] Initializing Edge-Case Sanity Check...")
    
    try:
        df = pd.read_csv("marts_telecom_features.csv").dropna(subset=['customer_action'])
    except FileNotFoundError:
        print("[!] ERROR: Dataset not found.")
        return

    df['customer_action'] = df['customer_action'].astype(int)
    cols_to_drop = ['username', 'purchaseid', 'purchasedate', 'province', 'salesagent', 'actual_paid_price']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop + ['customer_action'])
    y = df['customer_action']
    
    print("[*] Training Prescriptive XGBoost Engine...")
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    print("\n[*] Creating a Synthetic 'High-Risk / Angry' Customer...")
    # 1. Take an average customer baseline
    synthetic_user = X.mean().to_dict()
    
    # 2. Inject high-risk parameters (making the customer likely to churn)
    base_price = 100000.0  # Catalog price of 100k
    synthetic_user['catalog_price'] = base_price
    synthetic_user['arpu'] = base_price
    synthetic_user['hidden_discount_percentage'] = 0.0 # No current discount
    synthetic_user['hidden_discount_amount'] = 0.0
    
    # Simulating terrible service and dropping usage
    if 'disruptioncount' in synthetic_user:
        synthetic_user['disruptioncount'] = X['disruptioncount'].max() # Max network drops
    if 'arpu_trend' in synthetic_user:
        synthetic_user['arpu_trend'] = -50.0 # Dropping revenue trend
        
    user_df = pd.DataFrame([synthetic_user])
    
    print(f"    - Baseline Price: {base_price:,.0f}")
    print(f"    - Disruption Count: HIGH")
    print(f"    - Revenue Trend: NEGATIVE\n")
    
    # Setup simulation parameters
    alpha = 0.70
    discount_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    print("="*75)
    print(" DYNAMIC PRICING SIMULATION (RECALCULATING ELASTICITY)")
    print("="*75)
    print(f"{'Discount':<10} | {'P_Accept':<12} | {'P_Downgrade':<12} | {'P_Churn':<12} | {'Expected Value (EV)':<15}")
    print("-" * 75)
    
    best_ev = -1
    best_discount = 0
    
    for disc in discount_grid:
        # Update the features to reflect the proposed discount
        user_scenario = user_df.copy()
        
        # Injecting the discount directly into the model's feature space
        # (The model needs to "see" the discount to adjust probabilities)
        user_scenario['hidden_discount_percentage'] = disc * 100
        user_scenario['hidden_discount_amount'] = base_price * disc
        user_scenario['arpu'] = base_price * (1 - disc)
        
        offered_price = base_price * (1 - disc)
        
        # Predict NEW probabilities based on the discounted price
        probs = model.predict_proba(user_scenario)[0]
        p_acc, p_down, p_churn = probs[0], probs[1], probs[2]
        
        # Calculate EV
        ev = (p_acc * offered_price) + (p_down * offered_price * alpha)
        
        print(f"{disc*100:>5.0f}%     | %{p_acc*100:05.2f}       | %{p_down*100:05.2f}       | %{p_churn*100:05.2f}       | {ev:,.1f}")
        
        if ev > best_ev:
            best_ev = ev
            best_discount = disc
            
    print("="*75)
    print(f"[+] INTELLIGENT DECISION: Model recommends a {best_discount*100:.0f}% discount to maximize EV at {best_ev:,.1f}.")
    print("="*75)

if __name__ == "__main__":
    run_sanity_check()