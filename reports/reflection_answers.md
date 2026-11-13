1.Why is it important to understand the dataset before building an ML model?

Answer: "Garbage in, garbage out." If you build a model on messy, duplicate, or misunderstood data, the predictions will be wrong or useless. Understanding the data reveals missing values, outliers, and data types so we can clean and prepare the features properly.

2.Which table appears to be the central table in the Olist dataset? Why?

Answer: The orders table. It connects almost every other table: customers via customer_id, order items via order_id, payments via order_id, and reviews via order_id.

3.What is the difference between customer_id and customer_unique_id?

Answer: customer_id is created per purchase/order (transactional). customer_unique_id identifies the actual physical person. If one person buys 3 times on the platform, they will have 3 different customer_ids, but only 1 customer_unique_id.

4.Which columns appear to represent dates?

Answer: In the orders table: order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, and order_estimated_delivery_date. In reviews: review_creation_date and review_answer_timestamp.

5.Which columns appear to represent categorical variables?

Answer: order_status, payment_type, customer_state, customer_city, product_category_name, and seller_state.

6.Which columns contain missing values?

Answer: Delivery dates in orders (e.g., order_delivered_customer_date for undelivered or canceled orders), review comments/titles in reviews, and product_category_name in products.

7.Which tables can be joined using order_id?

Answer: orders, order_items, payments, and reviews.

8.Which tables can be joined using product_id?

Answer: order_items and products.

9.What are three potential Machine Learning problems that can be created from this dataset?

Answer:

Delivery Delay Prediction (Classification): Predict whether an order will arrive later than the estimated date (Late vs On Time).
Review Score Prediction (Regression / Classification): Predict customer rating (1 to 5 stars) based on delivery time, product category, and price.
Customer Segmentation / Churn (Clustering): Group customers by how frequently and how much they spend (RFM analysis).

10.What ethical concerns may arise when using customer data for ML?

Answer: Customer privacy (exposing full addresses or location coordinates), algorithmic bias (penalizing certain zip codes or regions with higher prices or refusal of service), and misuse of transaction history without user consent.