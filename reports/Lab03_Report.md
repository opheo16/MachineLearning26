# Laboratory Experiment 3: Data Quality Assessment, EDA, and Preprocessing

**Course:** Machine Learning  
**Unit:** Unit 1 – Mathematical Foundations, Data Engineering & Feature Engineering  
**Dataset:** Olist Brazilian E-Commerce Public Dataset (`olist_orders_abt.csv`)  
**Instructor:** Sharad Laad  
**Student Name:** Priyanshi Mali    
**Lab Number:** 3  
**Lab Title:** Data Quality Assessment, Exploratory Data Analysis, and Preprocessing  

---

## 1. Dataset Description

In this laboratory experiment, we performed a comprehensive Data Quality Assessment, Exploratory Data Analysis (EDA), and Preprocessing workflow on the Analytical Base Table (ABT) produced in Lab 2 (`olist_orders_abt.csv`).

* **Unit of Analysis:** One row represents one distinct customer order.
* **Initial Dimensions:** 99,441 rows × 29 columns.
* **Final Processed Dimensions:** 99,441 rows × 35 columns (with engineered indicators, log-transformed features, and ordinal categories).
* **Primary Objective:** Assess data integrity, detect anomalies, identify missingness patterns, resolve feature skewness, demonstrate scaling techniques, and prevent data leakage before model training in Lab 4.

---

## 2. Data Type Identification

Features in the Analytical Base Table were formally categorized into four statistical measurement levels to determine appropriate visualization and transformation strategies:

| Measurement Level | Description | Identified Columns in ABT |
| :--- | :--- | :--- |
| **Nominal** | Categorical variables without any intrinsic mathematical order or ranking. | `customer_state`, `customer_city`, `main_product_category`, `dominant_payment_type`, `order_status` |
| **Ordinal** | Categorical variables with a meaningful ranked order, where intervals between values are not strictly equal. | `review_score` (1 to 5), `delivery_speed_category` (Fast < Normal < Slow < Very Delayed), `payment_value_band` (Low < Medium < High < Premium) |
| **Interval** | Numerical variables where differences are meaningful, but zero does not represent the absence of the quantity. | `order_hour` (0–23), `order_month` (1–12), `order_day_of_week` (0–6) |
| **Ratio** | Numerical variables possessing a true, absolute zero point where ratios between values are physically meaningful. | `total_payment_value`, `total_price`, `total_freight`, `delivery_days`, `estimated_delivery_days`, `delivery_delay_days`, `total_items`, `unique_products`, `unique_sellers` |

---

## 3. Missing Value Analysis

We conducted systematic missingness auditing across all 99,441 records:

### Missingness Summary Table
| Feature | Missing Count | Missing Percentage (%) | Root Cause & Nature |
| :--- | :--- | :--- | :--- |
| `delivery_delay_days` | 2,965 | 2.98% | Orders canceled, in-transit, or undelivered. |
| `delivery_days` | 2,965 | 2.98% | Orders canceled, in-transit, or undelivered. |
| `delivery_speed_category`| 2,965 | 2.98% | Derived directly from `delivery_days`. |
| `main_product_category` | 2,185 | 2.20% | Uncategorized products in seller catalog. |
| `total_price` | 775 | 0.78% | Orders without item records in raw items table. |
| `unique_sellers` | 775 | 0.78% | Orders without item records in raw items table. |
| `total_freight` | 775 | 0.78% | Orders without item records in raw items table. |
| `total_items` | 775 | 0.78% | Orders without item records in raw items table. |
| `unique_products` | 775 | 0.78% | Orders without item records in raw items table. |
| `review_score` | 768 | 0.77% | Customers who did not submit a feedback survey. |
| `review_comment_count` | 768 | 0.77% | Customers who did not submit a feedback survey. |
| `has_review_comment` | 768 | 0.77% | Customers who did not submit a feedback survey. |
| `total_payment_value` | 1 | 0.001% | Missing transaction record in payment system. |
| `payment_types_count` | 1 | 0.001% | Missing transaction record in payment system. |
| `max_payment_installments`| 1 | 0.001% | Missing transaction record in payment system. |

### Interpretation Questions
* **Which feature has the highest missing percentage?**  
  `delivery_delay_days` and `delivery_days` (2.98%, 2,965 rows).
* **Should missing numerical values always be filled with the mean?**  
  No. When distributions contain heavy positive skewness and extreme outliers (such as monetary transactions), the arithmetic mean is significantly inflated. The **median** provides a robust, outlier-resistant central tendency measure.
* **Why does missing `delivery_days` carry operational meaning?**  
  A missing delivery date is Informative Missingness (Missing Not at Random). It signifies that fulfillment was interrupted, the parcel is currently en route, or the order was canceled prior to arrival. Dropping or naively zeroing these records destroys valuable business signals.
* **When should we create a missing-value indicator column?**  
  An indicator column (`missing_delivery_days = 1`) should be created when the very fact that data is missing correlates with the target outcome (e.g., undelivered orders correlate with low customer reviews or cancellation risks).

---

## 4. Univariate Analysis

Univariate analysis examined the individual distribution and frequency of each feature independently:

1. **Nominal Geographic Distribution (`customer_state`):**  
   Order frequencies are heavily concentrated in São Paulo (`SP`), followed by Rio de Janeiro (`RJ`) and Minas Gerais (`MG`). The southeast region of Brazil accounts for over 65% of all e-commerce activity, showing substantial geographic imbalance.
2. **Product Category Distribution (`main_product_category`):**  
   The market is led by top-ranking categories: `bed_bath_table`, `health_beauty`, `sports_leisure`, `furniture_decor`, and `computers_accessories`.
3. **Ordinal Delivery Distribution (`delivery_speed_category`):**  
   The majority of shipments fall into `Fast` (≤ 7 days) and `Normal` (8–15 days). `Slow` (16–30 days) and `Very Delayed` (> 30 days) comprise the long tail of fulfillment operations.
4. **Numerical Feature Distribution (`total_payment_value`):**  
   Highly right-skewed. The overwhelming majority of orders cluster between \$30 and \$180, while rare luxury or bulk commercial purchases create an extensive positive tail reaching \$13,664.08.

---

## 5. Multivariate Analysis

Multivariate exploration investigated cross-feature interactions and directional dependencies:

1. **Delivery Delay Days vs. Review Score:**  
   Scatter plotting and grouping reveal an unambiguous, steep inverse relationship. When `delivery_delay_days` is negative (delivered earlier than estimated), customers consistently award 4 or 5 stars. As soon as `delivery_delay_days` crosses into positive numbers (> 0 days delay), customer sentiment collapses immediately to 1-star reviews.
2. **Payment Value vs. Freight Cost colored by Late Delivery:**  
   Orders with unusually high freight values frequently coincide with remote geographic zones, which display a visibly higher probability of experiencing shipping delays (`is_late_delivery = 1`).
3. **Correlation Analysis:**  
   * Strong positive correlation ($r \approx 0.74$) between `total_price` and `total_payment_value`.
   * Moderate positive correlation ($r \approx 0.51$) between `delivery_delay_days` and `is_late_delivery`.
   * Moderate negative correlation ($r \approx -0.33$) between `delivery_delay_days` and `review_score`.

---

## 6. Outlier and Skewness Analysis

### Skewness Evaluation
| Feature | Skewness Coefficient | Direction | Action Taken |
| :--- | :--- | :--- | :--- |
| `total_freight` | 12.05 | Heavily Right-Skewed | Log Transformation (`np.log1p`) |
| `total_price` | 9.73 | Heavily Right-Skewed | Log Transformation (`np.log1p`) |
| `total_payment_value` | 9.15 | Heavily Right-Skewed | Log Transformation (`np.log1p`) |
| `total_items` | 7.53 | Right-Skewed | Retained as discrete count |
| `delivery_days` | 3.83 | Right-Skewed | Robust Imputation / Scaling |
| `delivery_delay_days` | 2.01 | Right-Skewed | Excluded from training (Leakage) |

### IQR Outlier Detection
Using the Interquartile Range ($IQR = Q_3 - Q_1$):
* **`total_payment_value`:**  
  $Q_1 = \$62.01$, $Q_3 = \$176.97$, $IQR = \$114.96$.  
  $\text{Upper Bound} = Q_3 + 1.5 \times IQR = \$349.41$.  
  Detected outliers: **7,866 rows (7.91%)**.
* **`delivery_days`:**  
  $Q_1 = 6.0\text{ days}$, $Q_3 = 15.0\text{ days}$, $IQR = 9.0\text{ days}$.  
  $\text{Upper Bound} = 15.0 + 1.5 \times 9.0 = 28.5\text{ days}$.  
  Detected outliers: **5,025 rows (5.05%)**.

**Should outliers be automatically deleted?**  
No. Outliers in e-commerce (such as high-value \$2,000+ purchases or long remote deliveries to the Amazon region) represent valid, high-impact business transactions rather than data entry errors. Deleting them blinds predictive models to high-value customers. Instead, they are stabilized using monotonic log transformations.

---

## 7. Data Cleaning and Preprocessing Strategy

| Problem Identified | Columns Affected | Treatment / Imputation Technique | Justification |
| :--- | :--- | :--- | :--- |
| **Missing Categorical Data** | `main_product_category`, `dominant_payment_type`, `delivery_speed_category`, `payment_value_band`, `customer_state`, `customer_city`, `order_status` | Imputed with `"Unknown"` | Preserves row count and avoids arbitrarily distorting known distribution frequencies. |
| **Missing Informative Dates** | `delivery_days` | Engineered binary indicator `missing_delivery_days` ($1$ if missing, $0$ otherwise) | Preserves the operational signal that the parcel was canceled or has not yet reached delivery status. |
| **Missing Numerical Values** | `total_payment_value`, `total_price`, `total_freight`, `delivery_days`, `estimated_delivery_days`, `delivery_delay_days`, `review_score`, `total_items`, `unique_products`, `unique_sellers`, etc. | Imputed with column **Median** | Median is immune to distortion from extreme skewness and high-value financial outliers. |
| **Heavy Positive Skewness** | `total_payment_value`, `total_freight`, `total_price` | Applied Logarithmic Transformation: $\ln(1 + x)$ via `np.log1p()` | Compresses extreme positive tails into near-normal bell distributions, improving linear and gradient model stability. |
| **Disparate Numerical Scales** | `total_items` (1–20) vs `total_payment_value` (\$10–\$13,000) | Evaluated `StandardScaler` ($z = \frac{x-\mu}{\sigma}$) and `RobustScaler` ($\frac{x-\text{median}}{IQR}$) | Prevents distance-based and gradient algorithms from being disproportionately dominated by large-scale features. |

---

## 8. Data Leakage Discussion

### Core Concept
> *"If a feature directly reveals the target, it is not intelligence. It is leakage."*

Data leakage occurs when an algorithm is trained using features containing information that would be physically impossible to possess at the point of prediction in an operational setting.

### Specific Leakage Scenario in Olist ABT
* **Target Variable:** `is_late_delivery` ($1$ if late, $0$ if on time).
* **Leaked Feature:** `delivery_delay_days` and `delivery_days`.
* **Mathematical Proof of Leakage:**
  $$\text{delivery\_delay\_days} = \text{actual\_delivery\_date} - \text{estimated\_delivery\_date}$$
  $$\text{is\_late\_delivery} = \begin{cases} 1, & \text{if } \text{delivery\_delay\_days} > 0 \\ 0, & \text{otherwise} \end{cases}$$
* **Why this destroys real-world machine learning:**  
  When an order is created, `actual_delivery_date` does not yet exist—it belongs to the future. If `delivery_delay_days` is included during training, the model achieves near-perfect artificial accuracy by simply testing if `delivery_delay_days > 0`. When deployed to predict late delivery on newly placed orders, the model completely fails because that column will be empty.
* **Excluded Leakage Columns:** `delivery_delay_days`, `delivery_speed_category`, and `delivery_days`.

---

## 9. Conclusion

In Laboratory Experiment 3, we successfully moved from raw tabular construction to production-grade data readiness:
1. Audited and identified nominal, ordinal, interval, and ratio columns.
2. Resolved missing values using robust median imputation, `"Unknown"` categoricals, and explicit missingness indicators.
3. Transformed highly skewed monetary metrics via logarithmic transformations.
4. Demonstrated the mechanics of `StandardScaler`, `MinMaxScaler`, and `RobustScaler`.
5. Strictly mapped and isolated target leakage features to safeguard real-world generalization.

The resulting dataset, **`data/processed/olist_orders_abt_cleaned.csv`** (99,441 rows × 35 columns), is fully preprocessed, verified, and ready for model training in Lab 4.
