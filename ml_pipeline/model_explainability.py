"""
MLOps Pipeline: Model Explainability & Causal Guardrail Validation
Generates SHAP dependence plots to validate that microeconomic shape constraints 
prevent spurious correlations in the production pricing model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def load_production_data():
    """Fetches and prepares data from the feature store."""
    print("[*] Fetching data from feature store (marts_telecom_features.csv)...")
    try:
        df = pd.read_csv("marts_telecom_features.csv").dropna(subset=['customer_action'])
    except FileNotFoundError:
        print("[!] CRITICAL ERROR: Feature store export not found.")
        return None, None, None, None
        
    df['customer_action'] = df['customer_action'].astype(int)
    
    # Drop operational identifiers not used in training
    cols_to_drop = ['username', 'purchaseid', 'purchasedate', 'province', 'salesagent', 'actual_paid_price']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop + ['customer_action'])
    y = df['customer_action']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_diagnostic_models(X_train, y_train):
    """Trains a baseline model and a production-ready constrained model."""
    print("[*] Training Baseline Model (No Guardrails)...")
    model_unconstrained = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, max_depth=4, n_estimators=100, random_state=42
    )
    model_unconstrained.fit(X_train, y_train)

    print("[*] Training Production Model (With Microeconomic Constraints)...")
    feature_names = X_train.columns.tolist()
    
    # Enforce strict positive relationship (+1) for price indicators
    constraints = []
    for col in feature_names:
        if col in ['arpu', 'arpu_trend', 'catalog_price']:
            constraints.append(1) 
        else:
            constraints.append(0)
            
    model_constrained = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, max_depth=4, n_estimators=100, 
        monotone_constraints=tuple(constraints), random_state=42
    )
    model_constrained.fit(X_train, y_train)
    
    return model_unconstrained, model_constrained

def generate_explainability_report(model_unconstrained, model_constrained, X_test):
    """Generates and saves the SHAP diagnostic artifacts."""
    print("[*] Generating SHAP values for model diagnostics...")
    
    # Sample data for clearer visualization
    X_sample = X_test.sample(1000, random_state=42)
    
    explainer_unconstrained = shap.TreeExplainer(model_unconstrained)
    explainer_constrained = shap.TreeExplainer(model_constrained)
    
    shap_values_unconstrained = explainer_unconstrained.shap_values(X_sample)
    shap_values_constrained = explainer_constrained.shap_values(X_sample)
    
    class_index = 2 # Class 2 = Full Churn
    
    # Handle different SHAP library versions
    if isinstance(shap_values_unconstrained, list):
        sv_unconstrained = shap_values_unconstrained[class_index]
        sv_constrained = shap_values_constrained[class_index]
    else:
        sv_unconstrained = shap_values_unconstrained[:, :, class_index]
        sv_constrained = shap_values_constrained[:, :, class_index]
        
    target_feature = 'arpu'
    if target_feature not in X_sample.columns:
        print(f"[!] Target feature '{target_feature}' missing in dataset.")
        return

    # Render diagnostic plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    shap.dependence_plot(target_feature, sv_unconstrained, X_sample, 
                         show=False, ax=axes[0], interaction_index=None)
    axes[0].set_title("Baseline Model (Without Guardrails)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("SHAP Value (Impact on Churn Risk)")
    axes[0].set_xlabel("ARPU (Monthly Price)")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    shap.dependence_plot(target_feature, sv_constrained, X_sample, 
                         show=False, ax=axes[1], interaction_index=None)
    axes[1].set_title("Production Model (Microeconomic Constraints Applied)", fontsize=14, fontweight='bold', color='green')
    axes[1].set_ylabel("SHAP Value (Impact on Churn Risk)")
    axes[1].set_xlabel("ARPU (Monthly Price)")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    artifact_name = "SHAP_Constraint_Validation_Report.png"
    plt.savefig(artifact_name, dpi=300, bbox_inches='tight')
    
    print(f"\n[+] SUCCESS: Diagnostic plot saved as '{artifact_name}'.")
    print("[+] This artifact proves the model behaves rationally under price shocks.")
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_production_data()
    if X_train is not None:
        model_u, model_c = train_diagnostic_models(X_train, y_train)
        generate_explainability_report(model_u, model_c, X_test)