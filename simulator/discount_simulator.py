"""
Dynamic Discounting & ROI Simulator
-----------------------------------
This module loads the pre-trained, calibrated 3-class XGBoost model and applies it 
to a cohort of users to find the optimal discount strategy (0% to 30%) that maximizes 
the Expected Value (EV).
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

def load_simulation_data(data_path, sample_size=2000):
    """Loads a sample of the data to simulate real-world API requests."""
    print(f"[*] Loading validation cohort from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("ERROR: Data file not found.")
        return None
        
    # Drop currently active grace-period users to simulate on historical closed cases,
    # or you can test on current active users if available.
    df_clean = df.dropna(subset=['customer_action']).copy()
    
    # Take a random sample to simulate a batch of users
    df_sim = df_clean.sample(n=min(sample_size, len(df_clean)), random_state=42).copy()
    df_sim['temp_user_id'] = range(len(df_sim))
    return df_sim

def load_model_artifacts(model_path, features_path):
    """Loads the pre-trained model and the exact feature list."""
    print("[*] Loading AI brain and feature schemas...")
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except FileNotFoundError:
        print("ERROR: Model artifacts not found. Please run train_multiclass.py first.")
        return None, None

def run_simulation(df_sim, model, feature_names):
    """Simulates multiple discount scenarios and predicts EV."""
    print("[*] Generating parallel universes (Applying discounts from 0% to 30%)...")
    
    # Reverse engineer base actual ARPU to calculate future trends correctly
    df_sim['prev_arpu_actual'] = df_sim['arpu'] - df_sim['arpu_trend']
    
    # Scenarios: 0, 5, 10, 15, 20, 25, 30 percent discount
    discount_scenarios = np.linspace(0.0, 0.30, 7)
    
    df_expanded = df_sim.loc[df_sim.index.repeat(len(discount_scenarios))].reset_index(drop=True)
    df_expanded['simulated_discount'] = np.tile(discount_scenarios, len(df_sim))
    
    # Re-engineer features for the AI based on the simulated discount
    df_expanded['hidden_discount_percentage'] = df_expanded['simulated_discount']
    df_expanded['offered_price'] = df_expanded['catalog_price'] * (1 - df_expanded['simulated_discount'])
    df_expanded['hidden_discount_amount'] = df_expanded['catalog_price'] - df_expanded['offered_price']
    
    df_expanded['arpu'] = np.where(df_expanded['duration'] > 0, df_expanded['offered_price'] / df_expanded['duration'], 0)
    df_expanded['arpu_trend'] = df_expanded['arpu'] - df_expanded['prev_arpu_actual']
    
    # Prepare exact features the model expects
    X_predict = df_expanded[feature_names].copy()
    
    print("[*] Predicting real-world probabilities using calibrated AI...")
    probs = model.predict_proba(X_predict)
    df_expanded['Prob_Accept'] = probs[:, 0]
    df_expanded['Prob_Downgrade'] = probs[:, 1]
    df_expanded['Prob_Churn'] = probs[:, 2]
    
    return df_expanded

def optimize_expected_value(df_expanded):
    """Finds the discount scenario with the highest financial expected value."""
    print("[*] Calculating Expected Value (EV)...")
    
    # Business logic: A downgraded customer yields roughly 70% of the offered price
    DOWNGRADE_REVENUE_RATIO = 0.70
    
    df_expanded['Expected_Value'] = (df_expanded['Prob_Accept'] * df_expanded['offered_price']) + \
                                    (df_expanded['Prob_Downgrade'] * (df_expanded['offered_price'] * DOWNGRADE_REVENUE_RATIO)) + \
                                    (df_expanded['Prob_Churn'] * 0)
                                    
    # Find the row (scenario) with maximum EV for each user
    optimal_scenarios = df_expanded.loc[df_expanded.groupby('temp_user_id')['Expected_Value'].idxmax()].copy()
    return optimal_scenarios

def generate_report(optimal_scenarios):
    """Prints the business ROI report and plots the strategy."""
    print("\n" + "="*60)
    print("     --- DYNAMIC DISCOUNTING ROI REPORT (MULTI-CLASS) ---")
    print("="*60)
    
    discount_distribution = optimal_scenarios['simulated_discount'].value_counts(normalize=True).sort_index() * 100
    
    print("\nAI Optimal Discount Strategy:")
    for discount, percentage in discount_distribution.items():
        print(f"- Recommended {int(discount*100)}% Discount for: {percentage:.1f}% of users")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f"{int(d*100)}%" for d in discount_distribution.index], discount_distribution.values, color='cornflowerblue', edgecolor='black')
    plt.title('AI-Recommended Targeted Discounts', fontsize=15, fontweight='bold')
    plt.xlabel('Suggested Discount', fontsize=12)
    plt.ylabel('Percentage of Users (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.show()

def main():
    # File paths
    data_path = "../marts_telecom_features.csv" # Adjust relative path based on execution folder
    model_path = "../ml_pipeline/calibrated_xgb_model.pkl"
    features_path = "../ml_pipeline/feature_names.pkl"
    
    # 1. Load Data
    df_sim = load_simulation_data(data_path, sample_size=2000)
    if df_sim is None: return
    
    # 2. Load Model
    model, feature_names = load_model_artifacts(model_path, features_path)
    if model is None: return
    
    # 3. Simulate and Predict
    df_expanded = run_simulation(df_sim, model, feature_names)
    
    # 4. Find Best Strategy
    optimal_scenarios = optimize_expected_value(df_expanded)
    
    # 5. Output Results
    generate_report(optimal_scenarios)

if __name__ == "__main__":
    # Note: To run this correctly, ensure your working directory is the 'simulator' folder
    main()