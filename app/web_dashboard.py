import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(page_title="AI Pricing Optimizer", page_icon="🚀", layout="wide")
st.title("🚀 Causal Dynamic Pricing & Retention Engine")
st.markdown("This AI-powered tool simulates price elasticity and causal downgrade risks to recommend the optimal discount for each at-risk customer, maximizing the company's Expected Value (EV).")

# --- Load AI Core ---
@st.cache_resource
def load_ai_brain():
    model_path = "ml_pipeline/calibrated_xgb_model.pkl"
    features_path = "ml_pipeline/feature_names.pkl"
    
    if not os.path.exists(model_path):
        model_path = "../ml_pipeline/calibrated_xgb_model.pkl"
        features_path = "../ml_pipeline/feature_names.pkl"
        
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except Exception as e:
        st.error("Error loading the AI model. Please ensure the pipeline has been executed.")
        return None, None

model, feature_names = load_ai_brain()

# --- Sample Data Guide (For Marketing Team) ---
with st.expander("💡 View Expected Data Format (Template)", expanded=False):
    st.markdown("Your uploaded CSV must be generated from the **dbt Data Warehouse**. It should contain the historical features of customers nearing their expiration date. Below is a structural example:")
    
    sample_df = pd.DataFrame({
        "username": ["user_101", "user_102", "user_103"],
        "catalog_price": [120000, 85000, 200000],
        "arpu": [120000, 80000, 180000],
        "arpu_trend": [0, -5000, 10000],
        "duration": [30, 30, 90],
        "rolling_3_disruptioncount": [1, 4, 0],
        "...": ["...", "...", "..."]
    })
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

# --- Sidebar & File Upload ---
st.sidebar.header("📥 Data Ingestion")
st.sidebar.info("Upload the CSV export containing at-risk customers (e.g., 5 days to expiration).")
uploaded_file = st.sidebar.file_uploader("Upload Target Cohort (CSV)", type=["csv"])

if uploaded_file is not None and model is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write(f"✅ Successfully loaded **{len(df_input):,}** customer records.")
    
    if st.button("🧠 Run AI Optimization", type="primary"):
        with st.spinner('AI is simulating parallel pricing universes...'):
            
            # --- Simulator Logic ---
            df_sim = df_input.copy()
            df_sim['temp_user_id'] = range(len(df_sim))
            
            if 'arpu_trend' in df_sim.columns and 'arpu' in df_sim.columns:
                df_sim['prev_arpu_actual'] = df_sim['arpu'] - df_sim['arpu_trend']
            else:
                st.error("CRITICAL ERROR: 'arpu' and 'arpu_trend' columns are missing!")
                st.stop()
                
            discount_scenarios = np.linspace(0.0, 0.30, 7)
            df_expanded = df_sim.loc[df_sim.index.repeat(len(discount_scenarios))].reset_index(drop=True)
            df_expanded['simulated_discount'] = np.tile(discount_scenarios, len(df_sim))
            
            df_expanded['hidden_discount_percentage'] = df_expanded['simulated_discount']
            df_expanded['offered_price'] = df_expanded['catalog_price'] * (1 - df_expanded['simulated_discount'])
            df_expanded['hidden_discount_amount'] = df_expanded['catalog_price'] - df_expanded['offered_price']
            
            df_expanded['arpu'] = np.where(df_expanded['duration'] > 0, df_expanded['offered_price'] / df_expanded['duration'], 0)
            df_expanded['arpu_trend'] = df_expanded['arpu'] - df_expanded['prev_arpu_actual']
            
            try:
                X_predict = df_expanded[feature_names]
            except KeyError as e:
                st.error(f"Missing required features in the uploaded dataset: {e}")
                st.stop()
                
            probs = model.predict_proba(X_predict)
            df_expanded['Prob_Accept'] = probs[:, 0]
            df_expanded['Prob_Downgrade'] = probs[:, 1]
            df_expanded['Prob_Churn'] = probs[:, 2]
            
            DOWNGRADE_REVENUE_RATIO = 0.70
            df_expanded['Expected_Value'] = (df_expanded['Prob_Accept'] * df_expanded['offered_price']) + \
                                            (df_expanded['Prob_Downgrade'] * (df_expanded['offered_price'] * DOWNGRADE_REVENUE_RATIO))
                                            
            optimal_scenarios = df_expanded.loc[df_expanded.groupby('temp_user_id')['Expected_Value'].idxmax()].copy()
            
            # --- Rendering Results ---
            st.success("🎉 ROI Optimization Complete!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 Discount Distribution")
                discount_distribution = optimal_scenarios['simulated_discount'].value_counts(normalize=True).sort_index() * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar([f"{int(d*100)}%" for d in discount_distribution.index], discount_distribution.values, color='cornflowerblue', edgecolor='black')
                ax.set_ylabel('Percentage of Cohort (%)')
                st.pyplot(fig)
                
            with col2:
                st.subheader("📋 Final Marketing Execution List")
                
                cols_to_show = ['username', 'arpu']
                if 'rolling_3_disruptioncount' in optimal_scenarios.columns:
                    cols_to_show.append('rolling_3_disruptioncount')
                    
                cols_to_show.extend(['catalog_price', 'simulated_discount', 'offered_price'])
                final_output = optimal_scenarios[cols_to_show].copy()
                
                final_output['simulated_discount'] = (final_output['simulated_discount'] * 100).astype(int).astype(str) + "%"
                final_output['arpu'] = final_output['arpu'].astype(int)
                final_output['catalog_price'] = final_output['catalog_price'].astype(int)
                final_output['offered_price'] = final_output['offered_price'].astype(int)
                
                rename_dict = {
                    'username': 'User ID',
                    'arpu': 'Current ARPU',
                    'rolling_3_disruptioncount': 'Recent Disruptions',
                    'catalog_price': 'Base Price',
                    'simulated_discount': 'AI Recommended Discount',
                    'offered_price': 'Final Offer Price'
                }
                final_output.rename(columns=rename_dict, inplace=True)
                
                st.dataframe(final_output.head(15), use_container_width=True)
                
            st.markdown("---")
            csv_data = final_output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Target List for SMS Campaign (CSV)",
                data=csv_data,
                file_name='marketing_ai_discounts.csv',
                mime='text/csv',
                type="primary"
            )
else:
    st.info("👈 Please upload the CSV cohort from the sidebar to begin.")