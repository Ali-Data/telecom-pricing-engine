"""
MLOps Pipeline: Financial Sensitivity & Strategy Diagnostics
Simulates total portfolio Expected Value (EV) across varying market response 
parameters (alpha) to evaluate the financial robustness of the dynamic pricing engine.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def run_financial_sensitivity():
    print("[*] Starting Financial Sensitivity Analysis...")
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train production model
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Create evaluation simulation space for X_test
    df_test = X_test.copy()
    df_test['temp_id'] = range(len(df_test))
    
    # Grid of alpha values to test market volatility (from 0.4 to 0.9)
    alpha_grid = np.linspace(0.40, 0.90, 6)
    discount_scenarios = np.linspace(0.0, 0.30, 7) # 0%, 5%, 10%, ..., 30%
    
    dynamic_ev_results = []
    no_discount_ev_results = []
    flat_discount_ev_results = []
    
    print("[*] Simulating parallel pricing universes across alpha spectrum...")
    
    for alpha in alpha_grid:
        # 1. Simulate Dynamic Pricing Strategy
        df_expanded = df_test.loc[df_test.index.repeat(len(discount_scenarios))].reset_index(drop=True)
        df_expanded['sim_discount'] = np.tile(discount_scenarios, len(df_test))
        df_expanded['offered_price'] = df_expanded['catalog_price'] * (1 - df_expanded['sim_discount'])
        
        # Predict probabilities for all parallel states
        # Note: In production, features would be dynamically updated; here we use base predictions for scaling
        probs = model.predict_proba(df_expanded.drop(columns=['temp_id', 'sim_discount', 'offered_price'], errors='ignore'))
        df_expanded['P_Acc'], df_expanded['P_Down'], df_expanded['P_Churn'] = probs[:, 0], probs[:, 1], probs[:, 2]
        
        # Apply the Portfolio EV Formula
        df_expanded['EV'] = (df_expanded['P_Acc'] * df_expanded['offered_price']) + \
                            (df_expanded['P_Down'] * df_expanded['offered_price'] * alpha)
                            
        # Extract the maximum EV achievable per customer
        optimal_per_user = df_expanded.loc[df_expanded.groupby('temp_id')['EV'].idxmax()]
        total_dynamic_ev = optimal_per_user['EV'].sum()
        dynamic_ev_results.append(total_dynamic_ev)
        
        # 2. Simulate "No Discount" Baseline (Flat 0% Discount)
        df_no_disc = df_expanded[df_expanded['sim_discount'] == 0.0]
        total_no_discount_ev = df_no_disc['EV'].sum()
        no_discount_ev_results.append(total_no_discount_ev)
        
        # 3. Simulate "Flat 15% Discount" Baseline (Traditional Campaign)
        df_flat_disc = df_expanded[np.isclose(df_expanded['sim_discount'], 0.15)]
        total_flat_ev = df_flat_disc['EV'].sum()
        flat_discount_ev_results.append(total_flat_ev)

    print("[*] Generating financial risk report plots...")
    # Plotting the Sensitivity Report
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_grid, dynamic_ev_results, marker='o', linewidth=2.5, color='green', label='Proposed: AI Dynamic Pricing Engine')
    plt.plot(alpha_grid, flat_discount_ev_results, marker='s', linestyle='--', color='orange', label='Baseline: Flat 15% Discount Campaign')
    plt.plot(alpha_grid, no_discount_ev_results, marker='x', linestyle=':', color='red', label='Baseline: Static Pricing (No Discount)')
    
    plt.title("Financial Sensitivity Analysis & Strategy Optimization", fontsize=14, fontweight='bold')
    plt.xlabel("Alpha (Revenue Retention Ratio under Package Downgrade)", fontsize=12)
    plt.ylabel("Total Portfolio Expected Value (Monetary Units)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    output_filename = "EV_Financial_Sensitivity_Report.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[+] SUCCESS: Financial diagnostic artifact saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    run_financial_sensitivity()