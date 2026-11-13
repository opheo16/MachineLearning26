# Laboratory Experiment 6: Feature Engineering & Feature Selection

**Course:** Machine Learning  
**Unit:** Unit 1 – Mathematical Foundations, Data Engineering & Feature Engineering  
**Lab Number:** 6  
**Lab Title:** Turning Data into ML Signals (Feature Engineering & Selection)  
**Instructor:** Sharad Laad  
**Student Name:** Priyanshi Mali

**Output Dataset:** `data/processed/olist_orders_feature_engineered.csv`

---

## 1. Executive Summary

In this laboratory experiment, we explored **Feature Engineering** and **Feature Selection** on the Olist e-commerce Analytical Base Table (ABT).

While Lab 2 focused on aggregating disparate tables into a unified observation format (one row per order), this lab focused on **improving the representation of that grain**:

1. Transforming raw business values into normalized ratios (e.g., `freight_ratio`, `average_item_price`).
2. Extracting temporal behavioral indicators (`is_weekend`, `is_business_hour`, `is_year_end`).
3. Encoding cyclical temporal loops using trigonometric transformations ($\sin$ and $\cos$).
4. Systematically filtering noise and multicollinearity through Variance Thresholds, Correlation Filtering, Mutual Information, and Tree Feature Importance.

---

## 2. Distinction: ABT Construction vs. Feature Engineering

| Dimension            | Lab 2: ABT Construction                                          | Lab 6: Feature Engineering                                            |
| :------------------- | :--------------------------------------------------------------- | :-------------------------------------------------------------------- |
| **Primary Goal**     | Combines information across multiple relational database tables. | Transforms information already consolidated in the ABT.               |
| **Analytical Grain** | Defines the grain (standardizes to one row per order).           | Improves the representation and expressiveness of that grain.         |
| **Methods Used**     | Table joins, `GROUP BY`, aggregations (`sum`, `count`, `first`). | Mathematical ratios, non-linear mappings, trigonometry, interactions. |
| **Core Question**    | _"What raw data belongs in one single row?"_                     | _"How can this data be reshaped into a stronger predictive signal?"_  |

---

## 3. Short Feature Engineering Report (Section 26)

| Item                                  | Student Response                                                                                                                              |
| :------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **Original number of features**       | 29 columns                                                                                                                                    |
| **Number of engineered features**     | 15 new features created                                                                                                                       |
| **Most useful engineered feature**    | `freight_ratio` ($\frac{\text{total\_freight}}{\text{total\_price}}$) and `estimated_delivery_days`                                           |
| **Why was it created?**               | A \$50 shipping fee is normal for a \$1,000 order, but huge for a \$20 order. `freight_ratio` normalizes shipping burden across order values. |
| **Features removed during selection** | `items_x_price` (redundant with `total_price`), identifier columns (`order_id`, `customer_id`), and post-order leakage columns.               |
| **Why were they removed?**            | Multicollinearity ($r > 0.90$) and prevention of target leakage.                                                                              |
| **Leakage risks found**               | Post-delivery columns (`delivery_days`, `delivery_delay_days`, `review_score`) must never be used to construct engineered features.           |
| **Baseline model result**             | Accuracy: ~72.1% \| F1: ~0.26                                                                                                                 |
| **Engineered model result**           | Accuracy: ~73.4% \| F1: ~0.29                                                                                                                 |
| **Final conclusion**                  | Feature engineering provided richer signals that improved class separation for late deliveries without adding data leakage.                   |

---

## 4. Challenge Exercise (Section 27)

### Feature 1: Freight Ratio

- **Original Features:** `total_freight`, `total_price`
- **New Feature:** `freight_ratio`
- **Formula:** $\text{total\_freight} / \text{total\_price}$
- **Business Interpretation:** The fraction of the purchase value spent purely on logistical delivery.
- **Why it helps ML:** Large freight ratios typically indicate heavy, bulky items or long transit distances to remote states, both of which strongly correlate with delayed shipping.
- **Potential Leakage Risk:** None; both price and freight are finalized and locked at checkout.

### Feature 2: Weekend Order Indicator

- **Original Feature:** `order_day_of_week`
- **New Feature:** `is_weekend`
- **Formula:** $1 \text{ if } \text{order\_day\_of_week} \ge 5 \text{ else } 0$
- **Business Interpretation:** Identifies orders placed on Saturday or Sunday.
- **Why it helps ML:** Courier facilities and seller dispatch hubs are often closed or operate with reduced staffing on weekends, delaying carrier pickup.
- **Potential Leakage Risk:** None; purchase timestamp is recorded immediately upon order submission.

### Feature 3: Cyclical Month Encoding

- **Original Feature:** `order_month`
- **New Features:** `month_sin`, `month_cos`
- **Formula:** $\sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$
- **Business Interpretation:** Preserves the cyclical nature of seasons and holidays (December and January are contiguous).
- **Why it helps ML:** Linear models otherwise treat Month 12 and Month 1 as maximally distant ($|12 - 1| = 11$). Sine/Cosine mappings position them next to each other on the unit circle.
- **Potential Leakage Risk:** None; order month is an immutable purchase property.

---

## 5. Student Reflection (Section 28)

- **Before Lab 6:** I thought a feature was simply any raw column that came out of the database or CSV file.
- **After Lab 6:** I now understand that a feature is a deliberate mathematical representation of a business phenomenon engineered specifically to help an algorithm detect patterns.
- **Most useful engineered feature:** `freight_ratio`, because it captures economic and logistical proportions far better than raw monetary values.
- **Most important lesson:** Feature engineering is an empirical exploration, not a magic formula. More features do not automatically mean higher accuracy; disciplined feature selection and strict leakage prevention are essential.

---

## 6. Conclusion & Key Takeaways

1. **ABT is not Feature Engineering:** ABT construction defines the subject grain; feature engineering creates signals on that grain.
2. **Trigonometric Encoding Resolves Cyclical Discontinuity:** 24-hour and 12-month cycles must be projected onto sine and cosine axes.
3. **Feature Selection Controls Complexity:** Eliminating near-zero variance features and highly correlated twins ($r > 0.90$) reduces dimensionality and prevents model overfitting.
4. **Leakage Vigilance:** An engineered feature built from future variables (e.g., delivery timestamps) is invalid and completely destroys real-world model deployment.
