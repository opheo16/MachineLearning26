# Laboratory Experiment 4: Scikit-Learn Preprocessing and Modeling Pipeline

**Course:** Machine Learning  
**Unit:** Unit 1 – Mathematical Foundations, Data Engineering & Feature Engineering  
**Lab Number:** 4  
**Lab Title:** Building a Leakage-Safe Scikit-Learn Preprocessing and Modeling Pipeline  
**Student Name:** Priyanshi Mali  Artifact Model Saved:** `models/late_delivery_pipeline.joblib`  

---

## 1. Executive Summary & Business Problem

In this experiment, we constructed an end-to-end, leakage-free Machine Learning system using Scikit-Learn Pipelines. 

* **Business Objective:** Predict whether an e-commerce order will experience a delayed delivery (`is_late_delivery = 1`) using only information available **at the time of purchase**.
* **Core Philosophy:** In production Machine Learning, data preprocessing (imputation, scaling, encoding) must never be performed manually or fitted across the entire dataset. By encapsulating preprocessing and model estimators into a unified `Pipeline`, we guarantee train-test consistency, prevent subtle data leakage, and ensure model reproducibility.

---

## 2. Dataset Dimensions & Target Distribution

* **Dataset Used:** Analytical Base Table (`olist_orders_abt.csv`)
* **Total Observations:** 99,441 customer orders
* **Target Variable:** `is_late_delivery` (Binary: $0 = \text{On-Time}$, $1 = \text{Late}$)

### Target Class Distribution Table
| Target Class | Meaning | Count (Orders) | Percentage (%) |
| :--- | :--- | :---: | :---: |
| **0** | Delivered On-Time or Early | ~91,894 | 92.41% |
| **1** | Delivered After Estimated Date (Late) | ~7,547 | 7.59% |
| **Total** | | 99,441 | 100.00% |

> **Observation:** The dataset is **heavily imbalanced** (~92% vs ~8%). Therefore, raw classification accuracy is a misleading performance metric; precision, recall, and F1-score must be evaluated.

---

## 3. Feature Selection & Leakage Prevention

To prevent artificial inflation of accuracy, non-predictive and post-event columns were removed prior to model training:

1. **Removed Identifier Columns (Cause Overfitting/Memorization):**
   * `order_id`
   * `customer_id`
   * `customer_unique_id`
   * `customer_city` (high cardinality with 4,000+ levels dropped to prevent feature explosion)

2. **Removed Data Leakage Columns (Contain Post-Delivery Information):**
   * `delivery_days`: Only recorded after delivery is completed.
   * `delivery_delay_days`: Directly computes the mathematical condition for `is_late_delivery`.
   * `delivery_speed_category`: Derived directly from `delivery_days`.
   * `missing_delivery_days`: Highly correlated with cancellation and non-delivery.
   * `review_score`, `review_comment_count`, `has_review_comment`, `is_low_review`: Surveys are answered after delivery.
   * `order_delivered_customer_date`: Post-fulfillment timestamp.

3. **Retained Feature Sets ($X$):**
   * **Numerical Features:** `order_year`, `order_month`, `order_day`, `order_day_of_week`, `order_hour`, `estimated_delivery_days`, `total_payment_value`, `max_payment_installments`, `payment_types_count`, `total_items`, `total_price`, `total_freight`, `unique_products`, `unique_sellers`.
   * **Categorical Features:** `customer_state`, `order_status`, `dominant_payment_type`, `main_product_category`.

---

## 4. Pipeline Architecture

We designed a two-branch `ColumnTransformer` integrated into an estimator pipeline:

```text
                     ┌── Numerical Features ──> SimpleImputer(median) ────> StandardScaler() ────────┐
Raw Order Data (X) ──┤                                                                              ├──> LogisticRegression / RandomForest
                     └── Categorical Features ─> SimpleImputer('Unknown') ─> OneHotEncoder(ignore) ──┘
```

* **Numerical Sub-Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
* **Categorical Sub-Pipeline:** `SimpleImputer(strategy='constant', fill_value='Unknown')` followed by `OneHotEncoder(handle_unknown='ignore')`.
* **Estimator:** `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`. Setting `class_weight='balanced'` penalizes mistakes on minority late deliveries proportionally to their frequency.

---

## 5. Model Evaluation Results

### Confusion Matrix
```text
                  Predicted On-Time (0)    Predicted Late (1)
Actual On-Time (0)        13,760                   4,619
Actual Late (1)              489                   1,020
```

### Classification Performance Table
| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **~74.3%** | Overall correct predictions across both classes. |
| **Recall (Class 1 - Late)** | **~67.6%** | The pipeline successfully detected over two-thirds of all late packages before delivery. |
| **Precision (Class 1 - Late)** | **~18.1%** | Due to heavy imbalance, catching 68% of late orders introduces false alarms. |
| **F1-Score (Class 1 - Late)** | **~0.29** | Harmonic mean of precision and recall for the rare class. |

---

## 6. Extension Task: Logistic Regression vs Random Forest

| Model | Overall Accuracy | Precision (Late) | Recall (Late) | F1-Score (Late) | Primary Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | ~74% | ~0.18 | **~0.68** | ~0.29 | High recall on rare late class; highly interpretable coefficients. |
| **Random Forest** | **~93%** | **~0.48** | ~0.18 | ~0.26 | High precision and high overall accuracy; captures non-linear feature interactions. |

* **Discussion:**
  1. **Which model performed better?** If the business priority is customer satisfaction through early intervention (catching late packages), **Logistic Regression** is superior due to high recall (~68%). If the business cannot afford false alarms (e.g., costly customer compensation), **Random Forest** is superior.
  2. **Which model is easier to interpret?** Logistic Regression is much easier to interpret because each feature has a direct linear odds multiplier.
  3. **Does Random Forest require scaling?** No, tree splits are invariant to monotonic numeric scaling, but the pipeline was preserved to handle categorical imputation and one-hot encoding seamlessly.

---

## 7. Answers to Student Reflection Questions

### Q1: What is the main purpose of a Scikit-Learn pipeline?
**Answer:** To bind preprocessing steps and estimators into a single cohesive object. This ensures identical transformations on train, test, and future production data while preventing data leakage.

### Q2: Why should preprocessing be fitted only on training data?
**Answer:** Fitting computes statistical parameters (like mean, median, standard deviation, and categorical dictionaries). If fitted on the whole dataset, test set characteristics leak into training, artificially inflating validation accuracy.

### Q3: What is the role of `ColumnTransformer`?
**Answer:** It applies distinct preprocessing pipelines to specific subsets of columns (numerical vs categorical) and concatenates the resulting matrices into a single unified feature array.

### Q4: Why do numerical and categorical columns need different preprocessing?
**Answer:** Numerical columns represent quantitative continuum values requiring imputation and scaling. Categorical columns represent discrete text labels that mathematical algorithms cannot process directly; they must be imputed with text categories and converted into binary flags via One-Hot Encoding.

### Q5: Why should identifiers not be used as features?
**Answer:** Identifier columns (`order_id`, `customer_id`) have unique, high-cardinality values with zero semantic pattern. Using them causes models to memorize specific training rows (overfitting) instead of learning generalizable relationships.

### Q6: Why is `delivery_delay_days` a leakage column?
**Answer:** `delivery_delay_days` is calculated as $\text{actual delivery date} - \text{estimated delivery date}$. Because actual delivery has not occurred when an order is placed, using this feature allows the model to "cheat" using future knowledge.

### Q7: Why is `OneHotEncoder(handle_unknown="ignore")` useful?
**Answer:** In production environments, newly added product categories or customer locations will inevitably appear. Setting `handle_unknown='ignore'` prevents the model from crashing by assigning zeros across all known category columns.

### Q8: Why did we use median imputation for numerical columns?
**Answer:** Business features such as order price, freight, and payment value are heavily right-skewed with extreme outliers. The median is a robust central tendency metric that is unaffected by extreme values, whereas the mean is distorted.

### Q9: Which algorithms are sensitive to scaling?
**Answer:** Distance-based algorithms ($k$-NN, SVM) and gradient-optimized algorithms (Logistic Regression, Linear Regression, Neural Networks) are sensitive to scaling. Tree-based algorithms (Decision Trees, Random Forests, XGBoost) are invariant to scaling.

### Q10: Why should we save the complete pipeline and not only the model?
**Answer:** A standalone trained model expects preprocessed, numerical, one-hot encoded matrices. If only the model is saved, raw incoming customer orders cannot be evaluated without rewriting all manual preprocessing code, introducing code divergence and deployment errors.

---

## 8. Conclusion

By building a leakage-safe pipeline, we completed our first functional supervised classification workflow:
1. Isolated target leakage and identifier features.
2. Created specialized numerical and categorical preprocessing pipelines.
3. Successfully trained, evaluated, and saved a persistent model artifact (`models/late_delivery_pipeline.joblib`).
4. Demonstrated end-to-end inference on raw sample order inputs.