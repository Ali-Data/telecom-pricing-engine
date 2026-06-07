{{ config(materialized='table') }}

WITH staging AS (
    SELECT 
        *,
        FlowCount AS SupportCallCount,
        CASE WHEN ProductName LIKE '%Static IP%' THEN 1 ELSE 0 END AS Has_Static_IP
    FROM {{ ref('stg_telecom_data') }}
),

catalog_pricing AS (
    SELECT 
        *,
        MAX(Price) OVER (
            PARTITION BY 
                DATE_TRUNC('month', PurchaseDate), 
                Duration, 
                Bandwidth, 
                Gig_Product, 
                Has_Static_IP
        ) AS Catalog_Price
    FROM staging
),

sequence_and_lags AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY Username ORDER BY PurchaseDate) AS Purchase_Sequence,
        COALESCE(SUM(Price) OVER (PARTITION BY Username ORDER BY PurchaseDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS Cumulative_LTV,
        
        LEAD(PurchaseDate, 1) OVER (w) AS Next_PurchaseDate,
        LEAD(Price, 1) OVER (w) AS Next_Price,
        LEAD(Duration, 1) OVER (w) AS Next_Duration,
        
        LAG(Price, 1) OVER (w) AS Prev_Price,
        LAG(Duration, 1) OVER (w) AS Prev_Duration,
        LAG(Bandwidth, 1) OVER (w) AS Prev_Bandwidth,
        LAG(Gig_Product, 1) OVER (w) AS Prev_Gig_Product,
        LAG(SupportCallCount, 1) OVER (w) AS Prev_SupportCallCount,
        
        AVG(DisruptionCount) OVER (PARTITION BY Username ORDER BY PurchaseDate ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS Rolling_3_DisruptionCount
    FROM catalog_pricing
    WINDOW w AS (PARTITION BY Username ORDER BY PurchaseDate)
),

feature_engineering AS (
    SELECT
        *,
        CASE WHEN Duration > 0 THEN Price / Duration ELSE 0 END AS Arpu,
        CASE WHEN Prev_Duration > 0 THEN Prev_Price / Prev_Duration ELSE 0 END AS Prev_Arpu,
        (Catalog_Price - Price) AS Hidden_Discount_Amount,
        CASE WHEN Catalog_Price > 0 THEN (Catalog_Price - Price) / Catalog_Price ELSE 0 END AS Hidden_Discount_Percentage,
        (PurchaseDate::DATE - Prev_ExpirationDate::DATE) AS Gap_Days,
        
        -- 3-Class Target Action Setup
        CASE 
            WHEN Next_PurchaseDate IS NOT NULL 
                 AND (Next_PurchaseDate::DATE - ExpirationDate::DATE) <= 15 
                 AND (CASE WHEN Next_Duration > 0 THEN Next_Price / Next_Duration ELSE 0 END) >= (CASE WHEN Duration > 0 THEN Price / Duration ELSE 0 END) 
                 THEN 0 -- Accept / Upgrade
            
            WHEN Next_PurchaseDate IS NOT NULL 
                 AND (Next_PurchaseDate::DATE - ExpirationDate::DATE) <= 15 
                 AND (CASE WHEN Next_Duration > 0 THEN Next_Price / Next_Duration ELSE 0 END) < (CASE WHEN Duration > 0 THEN Price / Duration ELSE 0 END) 
                 THEN 1 -- Downgrade
            
            WHEN (Next_PurchaseDate IS NOT NULL AND (Next_PurchaseDate::DATE - ExpirationDate::DATE) > 15)
                 OR (Next_PurchaseDate IS NULL AND CURRENT_DATE > (ExpirationDate::DATE + 15))
                 THEN 2 -- Full Churn
            ELSE NULL -- Censored data / Currently inside the Grace Period
        END AS Customer_Action
    FROM sequence_and_lags
)

SELECT
    Username, PurchaseID, PurchaseDate, Catalog_Price, Hidden_Discount_Amount, Hidden_Discount_Percentage,
    Duration, Bandwidth, Gig_Product, Has_Static_IP, Purchase_Sequence, Customer_Action, Arpu, Cumulative_LTV, Gap_Days,
    (Arpu - Prev_Arpu) AS Arpu_Trend,
    (SupportCallCount - Prev_SupportCallCount) AS SupportCallCount_Trend,
    Rolling_3_DisruptionCount
FROM feature_engineering;