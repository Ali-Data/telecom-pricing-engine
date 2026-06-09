# 🚀 AI-Driven Prescriptive Pricing & Churn Optimization Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![dbt](https://img.shields.io/badge/dbt-Data_Engineering-FF694B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

## 📌 Executive Summary
Traditional churn prediction models suffer from a fundamental flaw: they blindly identify *who* is leaving, leading to blanket promotional campaigns that cause severe **revenue cannibalization**. 

This repository contains an end-to-end **Prescriptive Machine Learning Pipeline** developed for the Telecommunications/ISP industry. Instead of binary churn prediction, this system acts as a financial guardrail. It utilizes **Theory-Guided Machine Learning (Microeconomic Shape Constraints)** and **Expected Value (EV) simulations** to dynamically allocate retention discounts only when mathematically justified, thereby maximizing overall portfolio ROI.

## 🏗️ System Architecture
The project is structured as a full-stack MLOps pipeline:

1. **Data Engineering Layer (dbt & SQL):**
   - Ingests raw transactional logs.
   - Uses advanced window functions to reverse-engineer marketing label noise.
   - Extracts underlying `catalog_price` and uncovers `latent_discount_percentages`.
2. **Predictive Layer (3-Class XGBoost):**
   - Moves beyond binary classification to a 3-state Markovian space: `Accept`, `Downgrade (Revenue Churn)`, and `Full Churn`.
   - Incorporates **Monotonic Constraints** to enforce rational economic behavior (+1 strict correlation between price shocks and churn risk), preventing spurious correlations.
   - Outputs are calibrated using **Isotonic Regression** to reflect true market probabilities.
3. **Prescriptive Layer (Financial Simulator):**
   - Runs parallel universe simulations across a grid of discount scenarios (0% to 30%).
   - Calculates the Expected Value (EV) for each scenario and prescribes the optimal individual discount to maximize corporate revenue.
4. **Interactive Dashboard (Streamlit):**
   - A real-time decision support system for marketing and pricing teams to analyze elasticity and trigger targeted retention campaigns.

---

# Causal Dynamic Pricing & Revenue Churn Optimization Engine (Telecom/ISP)

An end-to-end production-grade Data & Machine Learning pipeline that transitions a telecom provider from fixed static pricing to an enterprise-level **Hyper-Personalized Dynamic Discounting System**. 

This system models customer behavior in a 3-class paradigm (**Accept, Downgrade, Churn**), reverse-engineers hidden promotional discounts, enforces microeconomic rationality via causal monotonic constraints, and calibrates probabilities to safely maximize the company's Expected Value (EV) without destroying brand reputation.

---

## 🏗️ Repository Architecture

To mirror production environments, this repository separates data warehouse transformations from the machine learning orchestration layers.

    telecom-pricing-engine/
    ├── dbt_telecom/                         # Data Engineering & Transformations
    │   ├── models/                          # SQL models (staging, intermediate, marts)
    │   └── dbt_project.yml                  # dbt configuration
    ├── ml_pipeline/                         # MLOps & Diagnostic Suite
    │   ├── feature_robustness_test.py       # Ablation studies on latent features
    │   ├── financial_sensitivity_analysis.py# EV simulations under market volatility
    │   ├── model_explainability.py          # SHAP constraint validation plots
    │   ├── model_sanity_check.py            # Edge-case elasticity testing
    │   └── strategy_comparison_test.py      # Binary vs. 3-Class ROI benchmarking
    ├── app/                                 # Production UI
    │   └── streamlit_app.py                 # Interactive pricing dashboard
    ├── data/                                # Sample datasets (Git-ignored in production)
    ├── requirements.txt                     # Environment dependencies
    └── README.md                            # Project documentation

---

## 🛠️ MLOps Diagnostic Suite

To ensure the model is **robust**, **interpretable**, and **financially viable** before production deployment, this repository includes a comprehensive suite of diagnostic scripts in the `ml_pipeline/` directory:

- **`model_explainability.py`**  
  Validates the injection of microeconomic shape constraints using SHAP dependence plots. Proves the model successfully avoids spurious correlations.

- **`feature_robustness_test.py`**  
  Ablation study that quantifies the informational gain of reverse-engineered latent marketing variables.

- **`strategy_comparison_test.py`**  
  Benchmarks the financial ROI of the proposed Prescriptive AI against traditional Binary Churn models.

- **`financial_sensitivity_analysis.py`**  
  Simulates Total Portfolio Expected Value under varying market response parameters.

- **`model_sanity_check.py`**  
  Edge-case elasticity testing by injecting synthetic "high-risk" profiles.

---


## 🚀 How to Run the Pipeline

### 1. Environment Setup
    
    git clone <your-repository-url>
    cd telecom-pricing-engine
    pip install -r requirements.txt

### 2. Run Data Engineering (dbt)

    cd dbt_telecom
    dbt run
    cd ..
    
### 3. Run MLOps Diagnostics

    python ml_pipeline/model_explainability.py
    python ml_pipeline/strategy_comparison_test.py
    python ml_pipeline/feature_robustness_test.py
    python ml_pipeline/financial_sensitivity_analysis.py
    python ml_pipeline/model_sanity_check.py

### 4. Launch Production Dashboard
    Bash
    streamlit run app/streamlit_app.py

## 📚 Tech Stack
Data Engineering: dbt, SQL

Machine Learning: Python, XGBoost, Scikit-learn

Explainability: SHAP

Frontend: Streamlit
