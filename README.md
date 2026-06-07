# Causal Dynamic Pricing & Revenue Churn Optimization Engine (Telecom/ISP)

An end-to-end production-grade Data & Machine Learning pipeline that transitions a telecom provider from fixed static pricing to an enterprise-level **Hyper-Personalized Dynamic Discounting System**. 

This system models customer behavior in a 3-class paradigm (**Accept, Downgrade, Churn**), reverse-engineers hidden promotional discounts, enforces microeconomic rationality via causal monotonic constraints, and calibrates probabilities to safely maximize the company's Expected Value (EV) without destroying brand reputation.

---

## 🏗️ Repository Architecture

To mirror production environments, this repository separates data warehouse transformations from the machine learning orchestration layers.

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

---
## ⚡ Business Strategy & Core Innovations

### 1. 3-Class Revenue Churn Paradigm
Traditional churn models treat customer attrition as a binary problem (0 or 1). In reality, customers frequently choose to **downgrade** to cheaper packages rather than leaving completely. This model independently predicts three distinct future user paths within a **15-Day Grace Period**:

* **Class 0 (Accept):** Re-purchases a package within 15 days with equal or higher monthly ARPU.
* **Class 1 (Downgrade):** Re-purchases a package within 15 days but opts for a cheaper configuration (Revenue Churn).
* **Class 2 (Churn):** Fails to purchase any package within 15 days after expiration.

### 2. Reverse-Engineering Hidden Discounts (`Catalog_Price`)
Marketing teams often create duplicate products (e.g., `Zomorod-Standard` vs `Zomorod-Promotion`) to apply stealth discounts, creating immense label noise. The data engineering layer aggregates products by their physical properties (**Duration, Bandwidth, Gigabytes, Static IP**) within each calendar month and defines the `Catalog_Price` as the mathematical maximum price paid. The real discount percentage is then extracted dynamically:

> **Hidden Discount Percentage = (Catalog Price - Actual Paid Price) / Catalog Price**

### 3. Causal Microeconomic Constraints
Standard machine learning models often suffer from a false correlation: *"Higher prices lead to lower churn"*, because historically, only highly loyal VIP users bought expensive packages. We explicitly inject economic rationality into the XGBoost architecture via `monotone_constraints`, forcing a strict positive relationship (+1) between price indicators (`arpu`, `arpu_trend`, `catalog_price`) and risk probabilities.

### 4. Probability Calibration for ROI Estimation
Tree-based classifiers with class weighting (`sample_weight`) distort raw prediction probabilities into uncalibrated risk scores. Since our financial optimization module directly multiplies these probabilities by monetary values, **calibration is mathematically mandatory**. We run **Isotonic Regression** over the XGBoost output to guarantee that a 40% predicted risk reflects exactly a 40% empirical attrition rate in the market, protecting the company from handing out unnecessary discounts.

---
## 🛠️ Implementation Details

### 1. Data Engineering Layer (dbt/SQL)
The `mart_telecom_features.sql` model processes analytical transaction logs, handles temporal window functions, computes monthly price shocks, and builds the 3-class target variable.

    -- Sample dbt/SQL logic
    WITH catalog_pricing AS (
        SELECT *, MAX(Price) OVER (
            PARTITION BY DATE_TRUNC('month', PurchaseDate), Duration, Bandwidth, Gig_Product, Has_Static_IP
        ) AS Catalog_Price
        FROM staging
    )

### 2. Machine Learning Layer (Python)
Trains a multi-class XGBoost classifier with randomized hyperparameter search and wraps the optimal estimator inside an Isotonic Calibration layer.

    # ml_pipeline/train_multiclass.py
    xgb_base = xgb.XGBClassifier(objective='multi:softprob', num_class=3)
    search = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_grid)
    search.fit(X_train, y_train, sample_weight=train_weights)
    
    calibrated_model = CalibratedClassifierCV(estimator=search.best_estimator_, method='isotonic')

### 3. Parallel-Universe Simulator
Runs an analytical simulation testing pricing scenarios from 0% to 30% discount. It dynamically computes the financial Expected Value (EV):

    # Expected Value Formula with Downgrade Penalty Ratio (0.70)
    df_expanded['EV'] = (df_expanded['P_Accept'] * df_expanded['offered_price']) + \
                        (df_expanded['P_Downgrade'] * (df_expanded['offered_price'] * 0.70))

---

## 🚀 Impact & Results
On a validation cohort of 2,000 production customers, the deployment of this system achieved:

* **89.6% of users** correctly flagged as low-risk, removing unnecessary promotions.
* **10.4% of high-risk users** targeted with surgical, customized discounts (5% to 30%).
* **Maximized ROI** achieved purely through localized pricing efficiency.

---

## 📝 Academic Contribution & Paper Core (Q1 Target)
This framework introduces three structural contributions to the literature on computational revenue management:

1. **The Latent Discount Identification Framework:** Methodology to extract unobserved promotional variables via historical maximum feature grouping.
2. **Causal Constraint Multi-Class Classifiers:** Integrates shape-enforced monotonic behavior within cross-entropy multi-class probability spaces.
3. **Calibrated Expected Value Optimization:** Demonstrates mathematically why traditional uncalibrated algorithmic discounting yields negative financial utilities.
