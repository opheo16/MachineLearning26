# Laboratory Experiment 2: Building an Analytical Base Table for Machine Learning

**Course:** Machine Learning  
**Unit:** Unit 1 – Mathematical Foundations, Data Engineering & Feature Engineering  
**Dataset:** Olist Brazilian E-Commerce Public Dataset  
**Instructor:** Sharad Laad  
**Student Name:**  Priyanshi Mali  


---

## 1. Executive Summary

In this laboratory experiment, we converted raw relational e-commerce data spread across multiple disconnected tables into a single unified **Analytical Base Table (ABT)**. 

Machine learning algorithms require a structured rectangular dataset where each row represents one distinct observation and columns represent descriptive features or target outcomes. By standardizing the unit of analysis to **one row per order**, performing necessary aggregations on one-to-many relationships (payments, items, and reviews), engineering temporal and logistical features, and defining binary target variables, we produced a clean dataset ready for machine learning tasks.

* **Output Dataset:** `data/processed/olist_orders_abt.csv`
* **Final Dimension:** 99,441 rows × 29 columns
* **Summary Metadata Report:** `reports/lab02_abt_summary.csv`
* **Duplicate Orders:** 0 (verified primary key integrity)

---

## 2. Conceptual Questions & Answers

### Q1: What is an Analytical Base Table (ABT)?
**Answer:**  
An Analytical Base Table (ABT) is a consolidated, clean, tabular dataset created by integrating and transforming raw data from multiple relational sources into a single matrix. In an ABT, each row corresponds to a single subject of interest (in our case, an order), while columns contain engineered numerical, categorical, and target variables prepared specifically for machine learning models.

---

### Q2: Why did we choose one row per order for this lab?
**Answer:**  
In e-commerce operations, key business decisions and predictions—such as whether a delivery will arrive late or whether a customer will be dissatisfied—occur at the level of the entire order. Choosing one row per order establishes a consistent unit of analysis, ensuring that predictions and performance metrics are unambiguous and not duplicated.

---

### Q3: Why should we aggregate `payments` before merging with `orders`?
**Answer:**  
A single customer order can involve multiple payment methods (e.g., split across two credit cards or vouchers), representing a one-to-many relationship. If we merged the raw `payments` table directly without aggregation, orders with multiple payment rows would multiply into several rows. Aggregation summarizes these multiple records into single values per order (such as total payment value, maximum installments, and dominant payment method).

---

### Q4: Why should we aggregate `order_items` before merging with `orders`?
**Answer:**  
An order can contain multiple items. Merging `order_items` directly into `orders` would cause row duplication proportional to the number of items purchased in that order. By aggregating first, we compress the item information into concise order-level metrics: total items, total order price, total freight cost, unique product counts, and primary product category.

---

### Q5: What can go wrong if we directly merge one-to-many tables without aggregation?
**Answer:**  
Directly merging one-to-many tables creates **row explosion** (multiplication of identical order records). This creates severe issues:
1. It artificially inflates the dataset size.
2. It introduces duplicate target labels for the same order, creating data bias and distorting evaluation metrics.
3. It breaks the fundamental requirement of machine learning algorithms that each row represents an independent observation.

---

### Q6: Which features may help predict late delivery?
**Answer:**  
Features available at the time of purchase that can predict late delivery include:
* `estimated_delivery_days`: The promised delivery window.
* `customer_state`: Geographical transit distance and regional logistics infrastructure.
* `total_items` and `total_freight`: Larger, heavier orders frequently experience longer fulfillment and handling times.
* `order_month` / `order_day_of_week`: Peak shopping periods (e.g., Black Friday, holidays) and weekend processing bottlenecks.

---

### Q7: Which features may help predict a low review score?
**Answer:**  
* `is_late_delivery` and `delivery_delay_days`: Operational delays are the strongest determinant of customer dissatisfaction.
* `delivery_days`: Total transit duration experienced by the customer.
* `total_freight`: Disproportionately high shipping charges often lead to negative reviews.
* `dominant_payment_type` & `max_payment_installments`: High-value installment purchases often carry higher customer expectations.

---

### Q8: What is the difference between `review_score` and `is_low_review`?
**Answer:**  
* `review_score` is a **continuous/numerical** rating scale (ranging from 1 to 5) suitable for regression modeling.
* `is_low_review` is a **binary categorical target variable** where ratings of 1 or 2 are encoded as `1` (low satisfaction/negative review) and ratings of 3, 4, or 5 are encoded as `0` (positive/acceptable review). This reformulates the problem into binary classification.

---

### Q9: Why is `is_late_delivery` a classification target?
**Answer:**  
`is_late_delivery` is a classification target because its output space consists of discrete, mutually exclusive classes: `1` (the shipment arrived after the estimated delivery date) and `0` (the shipment arrived on or before the estimated date). Predicting categorical outcomes rather than continuous quantities is the defining characteristic of classification.

---

### Q10: Which columns should not be used directly as ML features? Why?
**Answer:**  
1. **Identifier Columns** (`order_id`, `customer_id`, `customer_unique_id`): These are high-cardinality arbitrary identifiers that carry no predictive signal and cause models to overfit.
2. **Leaked/Future Columns** (`delivery_days`, `delivery_delay_days`, `review_score`): These values are recorded only after the transaction and delivery are complete. Using them to predict outcomes at the time an order is placed constitutes **Target Leakage**, resulting in unrealistically high training performance that completely fails in actual production deployment.

---

## 3. Reflection Activity

### Topic: How does building an Analytical Base Table help convert business data into Machine Learning-ready data?

> Real-world enterprise data is organized across normalized relational database schemas (customers, orders, items, payments, reviews) optimized for transactional integrity and storage efficiency, rather than analytical consumption. Building an Analytical Base Table (ABT) bridges this fundamental gap by systematically resolving complex table relationships and aggregating one-to-many entities into a standardized "one row per observation" structure. In this process, raw timestamps and transaction logs are transformed through feature engineering into actionable indicators—such as cyclical date components, total financial metrics, and delivery durations. Concurrently, explicit supervised target variables (e.g., `is_late_delivery` and `is_low_review`) are formalized to define the predictive objective. By removing redundant identifiers, preventing row explosion, handling missing values, and strictly isolating potential target leakage, the ABT constructs a reliable, uniform, and robust foundation that enables machine learning algorithms to train effectively and generalize accurately to unseen data.

---

## 4. Key Engineered Features & Target Summary

| Feature Category | Column Names | Description |
| :--- | :--- | :--- |
| **Identifiers** | `order_id`, `customer_id`, `customer_unique_id` | Tracking keys (excluded from direct model inputs). |
| **Location** | `customer_city`, `customer_state` | Geographic attributes of the customer. |
| **Temporal** | `order_year`, `order_month`, `order_day`, `order_day_of_week`, `order_hour` | Extracted purchase timing components. |
| **Delivery Metrics** | `delivery_days`, `estimated_delivery_days`, `delivery_delay_days` | Fulfillment timeline metrics. |
| **Payments** | `total_payment_value`, `max_payment_installments`, `payment_types_count`, `dominant_payment_type` | Aggregated financial indicators per order. |
| **Order Items** | `total_items`, `total_price`, `total_freight`, `unique_products`, `unique_sellers`, `main_product_category` | Aggregated basket metrics and primary item category. |
| **Review Features** | `review_comment_count`, `has_review_comment` | Customer feedback activity flags. |
| **Target Variables** | `is_late_delivery`, `is_low_review` | Binary targets ($0$ or $1$) for classification models. |

---

## 5. Artifacts and Generated Visualizations

The following deliverables were generated and verified:
1. **`figures/lab02_late_delivery_distribution.png`**: Visualizes the class balance between on-time and late deliveries.
2. **`figures/lab02_review_score_distribution.png`**: Displays customer ratings distribution across the 1–5 scale.
3. **`figures/lab02_payment_value_distribution.png`**: Histogram showcasing the right-skewed distribution of order monetary values.
4. **`figures/lab02_top_customer_states.png`**: Highlights the geographical concentration of orders across top Brazilian states (e.g., SP, RJ, MG).
5. **`reports/lab02_abt_summary.csv`**: Automated data dictionary reporting feature data types, missing counts, null percentages, and unique value counts.
6. **`data/processed/olist_orders_abt.csv`**: The complete processed Analytical Base Table.