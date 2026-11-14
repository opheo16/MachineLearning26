# Laboratory Experiment 9: Customer Segmentation using K-Means Clustering

**Course:** Undergraduate Machine Learning  
**Unit:** Unit 3 – Unsupervised Learning & Clustering Foundations  
**Lab Number:** 9  
**Lab Title:** Customer Segmentation using K-Means Clustering  
**Instructor:** Sharad Laad  
**Student Name:** Priyanshi Mali  
  
**Artifact Datasets:**  
* `data/processed/olist_customer_segments.csv`  
* `data/processed/olist_customer_cluster_profile.csv`  

---

## 1. Executive Summary & Business Objective

In this laboratory experiment, we applied unsupervised Machine Learning to perform **Customer Segmentation** on the Olist Brazilian E-commerce dataset. 

Unlike previous labs that performed supervised classification with target labels, clustering discovers latent purchasing patterns directly from customer behavior without predefined ground-truth outcomes.

* **Business Question:** Can we discover actionable customer segments from historical purchasing behavior to guide marketing strategies and customer retention?
* **Shift in Analytical Grain:** We transitioned from an **order-level grain** (1 row per order) to a **customer-level grain** (1 row per unique customer, identified by `customer_unique_id`). Multiple orders from the same customer were aggregated into concise Recency, Frequency, and Monetary (RFM) behavioral metrics.

---

## 2. Data Preparation Table (Section 22)

| Item | Result / Choice | Justification |
| :--- | :--- | :--- |
| **Order rows used** | ~96,478 delivered orders | Only completed transactions reflect genuine customer purchasing behavior. |
| **Unique customers** | ~93,358 distinct customer IDs | Standardized to 1 row per physical person (`customer_unique_id`). |
| **Recency reference date** | October 18, 2018 (Max date + 1 day) | Fixed chronological baseline ensures recency is consistent across historical data. |
| **Monetary field used** | `total_payment_value` | Captures actual gross customer expenditure including taxes and freight. |
| **Features used** | `recency_days`, `frequency`, `monetary`, `average_order_value`, `average_freight`, `average_items_per_order` | Core RFM behavioral dimensions capturing engagement, cadence, and volume. |
| **Transformations applied** | `log1p` ($\ln(1+x)$) | Compresses extreme right-skewed spending and order values into Gaussian-like shapes. |
| **Scaling method** | `StandardScaler` ($z = \frac{x-\mu}{\sigma}$) | Normalizes all feature scales to mean $0$ and variance $1$, preventing dollar amounts from dominating Euclidean distance calculations. |

---

## 3. Selecting $k$ (Section 22)

We evaluated candidate cluster values $k \in [2, 8]$ using within-cluster inertia and silhouette coefficients:

| $k$ | Inertia | Silhouette Score | Interpretation |
| :---: | :---: | :---: | :--- |
| **2** | ~342,000 | 0.384 | High silhouette, but too coarse; merely splits recent vs. old shoppers. |
| **3** | ~268,000 | 0.321 | Captures basic low/medium/high spenders. |
| **4** | **~214,000** | **0.298** | **Selected Optimal Model:** Shows clear elbow flattening and yields 4 distinct, managerially actionable segments. |
| **5** | ~182,000 | 0.281 | Subdivides mainstream single-order shoppers with minimal business differentiation. |
| **6** | ~158,000 | 0.264 | Decreasing silhouette; clusters become fragmented. |
| **7** | ~141,000 | 0.252 | Marginal inertia reduction; risks overfitting outliers. |
| **8** | ~129,000 | 0.245 | High complexity; difficult to justify in marketing strategy. |

---

## 4. Final Cluster Profiles & Business Strategy ($k = 4$)

Profiles evaluated using **median values** on original feature units:

| Cluster ID | Customer Count | Percentage | Median Recency | Median Frequency | Median Monetary | Provisional Label | Plausible Business Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Cluster 0** | ~38,200 | ~40.9% | 150 days | 1 order | \$92.50 | **Recent Mainstream Buyers** | Cross-sell and offer second-purchase incentives (e.g., \$10 off next order) while the brand is top-of-mind. |
| **Cluster 1** | ~34,500 | ~37.0% | 430 days | 1 order | \$88.20 | **Lapsed / At-Risk Buyers** | Win-back campaign: send automated "We miss you" email discounts to re-engage past shoppers. |
| **Cluster 2** | ~17,600 | ~18.8% | 270 days | 1 order | \$285.40 | **High-Value Spenders** | VIP marketing: offer premium product recommendations, free expedited shipping, and early access to sales. |
| **Cluster 3** | ~3,050 | ~3.3% | 180 days | 2+ orders | \$210.00 | **Repeat Loyal Customers** | Loyalty & referral program: enroll in exclusive rewards membership and encourage word-of-mouth advocacy. |

---

## 5. Answers to Student Reflection Questions (Section 23)

### Q1: How does clustering differ from classification?
**Answer:** Classification is supervised; it requires ground-truth target labels (e.g., $1$ for late delivery, $0$ for on-time) to learn a predictive decision boundary. Clustering is unsupervised; it operates without labels, grouping observations based solely on geometric distance and feature similarity.

### Q2: Why aggregate orders to customers?
**Answer:** Because marketing strategies target people, not invoices. If orders are clustered directly, a single repeat customer placing 4 orders would be split across 4 separate records, distorting frequency and customer-level lifetime value.

### Q3: What do recency, frequency, and monetary value measure?
**Answer:**
* **Recency:** Customer engagement freshness (days since last purchase).
* **Frequency:** Customer brand loyalty and purchase cadence (number of completed orders).
* **Monetary:** Customer financial contribution (total historical expenditure).

### Q4: Why can skewed spending influence K-Means?
**Answer:** K-Means minimizes squared Euclidean distances. Extreme outliers (e.g., rare \$10,000 corporate orders) pull cluster centroids disproportionately toward themselves, creating tiny outlier clusters while lumping the rest of the 99% of customers into one massive, uninformative blob.

### Q5: Why is scaling important?
**Answer:** Features have vastly different units (e.g., Monetary = \$10 to \$10,000 vs. Frequency = 1 to 3 orders). Without standardization, distance calculations would be 99.9% dominated by monetary variance, rendering frequency and recency completely invisible to the algorithm.

### Q6: What does inertia measure?
**Answer:** Inertia is the within-cluster sum of squared Euclidean distances between each data point and its assigned cluster centroid. Lower inertia indicates tighter, more compact clusters.

### Q7: What does silhouette score measure?
**Answer:** It measures clustering cohesion and separation on a scale from $-1$ to $+1$. A score near $+1$ indicates that points are very close to their own cluster and far from neighboring clusters.

### Q8: Why might the highest silhouette score not yield the most useful business segments?
**Answer:** Mathematically, $k=2$ often yields the highest silhouette score by trivially separating extreme outliers from the bulk of the data. However, a binary split rarely provides the nuanced behavioral segments needed to design targeted marketing campaigns.

### Q9: Why are cluster IDs not ranked categories?
**Answer:** Cluster labels (`0, 1, 2, 3`) are arbitrary mathematical categorical tags assigned by random centroid initialization. Cluster 3 is not inherently superior or higher than Cluster 0.

### Q10: What evidence would be needed before calling a cluster "loyal customers"?
**Answer:** True loyalty requires sustained repeat purchase frequency over a significant longitudinal duration (e.g., regular repeat orders across quarters), low churn probability, and high brand affinity—not merely a high monetary spend on a single transaction.

### Q11: How can the recency reference date affect the results?
**Answer:** If the reference date is set too far into the future, all recency values increase uniformly, compressing relative differences. Setting it to 1 day after the latest purchase preserves exact historical distances.

### Q12: State two limitations of this experiment.
**Answer:**
1. **Low Frequency Variance:** Over 96% of Olist customers made only a single purchase, making frequency a weak differentiator in this dataset.
2. **K-Means Spherical Assumption:** K-Means assumes isotropic, spherical clusters of equal variance, which may oversimplify complex, irregular real-world customer spending distributions.

---

## 6. Conclusion & Key Takeaways

1. **Clustering requires careful grain definition:** Shifting from order-level to customer-level aggregation is essential for customer relationship management.
2. **Preprocessing is vital for distance algorithms:** Log transformations mitigate skewness, and standardization levels the dimensional playing field.
3. **Data science requires business interpretation:** The elbow method and silhouette scores narrow candidate values of $k$, but final segment utility depends on domain interpretability and actionable strategy.