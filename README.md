# Causal Dynamic Pricing & Revenue Churn Optimization Engine (Telecom/ISP)

An end-to-end production-grade Data & Machine Learning pipeline that transitions a telecom provider from fixed static pricing to an enterprise-level **Hyper-Personalized Dynamic Discounting System**. 

This system models customer behavior in a 3-class paradigm (**Accept, Downgrade, Churn**), reverse-engineers hidden promotional discounts, enforces microeconomic rationality via causal monotonic constraints, and calibrates probabilities to safely maximize the company's Expected Value (EV) without destroying brand reputation.

---

## 🏗️ Repository Architecture

To mirror production environments, this repository separates data warehouse transformations from the machine learning orchestration layers.

```text
├── dbt_telecom/                  # Data Engineering Layer (dbt Project)
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_telecom_data.sql
│   │   └── marts/
│   │       └── mart_telecom_features.sql
├── ml_pipeline/                  # Machine Learning Layer
│   ├── train_multiclass.py       # Hyperparameter tuning & XGBoost training
│   └── calibration.py            # Isotonic Regression probability calibration
├── simulator/                    # Decision Optimization Layer
│   └── discount_simulator.py     # Parallel-universe simulation & EV optimization
├── app/                          # Business Interface
│   └── web_dashboard.py          # Streamlit UI for the Marketing Team
├── requirements.txt              # Dependency management
└── README.md                     # Documentation (This file)
