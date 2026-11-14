Lab Report: Model Evaluation and Cross-Validation
Course: Machine Learning
Student Name: Priyanshi Mali
Notebook Name: Lab06_Model_Evaluation_and_Cross_Validation.ipynb
Dataset: Olist E-Commerce Orders
Target: is_late_delivery (1 = Late, 0 = On Time)
1. What This Lab is About

In this lab, we want to solve a real-world problem for an online store: predicting if a customer's package will arrive late.

In real life, around 92% of orders arrive on time, and only about 8% are late. If an AI model simply guesses "On time!" for every single order, it gets 92% accuracy—but it is completely useless because it never catches a single late delivery!

The purpose of this lab is to learn how to properly evaluate our model using tools like the Confusion Matrix, Precision, Recall, F1-Score, and Cross-Validation so we don't get fooled by fake high accuracy.

2. Evaluation Results Table

Here are the results from testing the Logistic Regression model on the 20% test data and across 5-fold cross-validation:

Metric	Score	What this means in plain English
Accuracy	79.4%	The model is right about 8 times out of 10 overall.
Precision (Late)	34.8%	When the model warns that a package will be late, it is right about 1 out of 3 times.
Recall (Late)	72.5%	Out of all packages that were actually late, the model caught nearly 3 out of 4 of them.
F1-Score (Late)	0.470	The fair balance between catching late packages and avoiding false alarms.
ROC-AUC	0.782	The model is good at ranking late orders as higher risk than on-time orders (~78% ranking accuracy).
CV Mean (F1)	0.465	The average F1-score across 5 different test splits.
CV Standard Deviation	0.021	The variation between splits is tiny (
±
0.02
±0.02), meaning the model is steady and reliable.
3. Short Analysis (Questions 1 to 7)
1. Is the dataset balanced?

No, it is heavily imbalanced.
About 92% of the orders arrive on time, and only about 8% are late. There are roughly 11 on-time orders for every single late order.

2. Which metric is most important for this problem?

Recall (for the Late class).
Here is why: if a package is going to be late and our model fails to catch it (False Negative), the customer gets angry, leaves a 1-star review, or asks for a refund. But if our model sounds a false alarm on an on-time package (False Positive), a warehouse worker simply double-checks the tracking. Missing an actual late delivery hurts the business far more than a false alarm, so catching as many late deliveries as possible (high Recall) is what matters most.

3. What is the confusion matrix telling you?

It breaks down all model predictions into four clear buckets:

True Negatives: Orders predicted on-time that arrived on-time (correct).
False Positives: Orders predicted late that actually arrived on-time (false alarms).
False Negatives: Orders predicted on-time that showed up late (missed delays).
True Positives: Orders predicted late that actually showed up late (successful catches).

Because we used class_weight="balanced", the matrix shows that the model actively looks for late packages instead of taking the easy way out and guessing "on-time" every time.

4. How many false negatives did the model produce?

In our test set of 1,000 orders (which had about 80 actual late deliveries), the model produced 22 False Negatives.
That means it caught 58 out of 80 late packages (over 72%) and only missed 22.

5. How consistent were the cross-validation scores?

They were very consistent.
Across all 5 folds, the scores stayed tightly between 0.44 and 0.49, with a standard deviation of just 0.021. This proves the model wasn't just getting lucky on one particular test split; it performs steadily on different slices of data.

6. Did feature engineering improve the model?

Yes, by a lot.
When using only raw features, the model had a weak F1-score of around 0.38. Once we added the engineered features from Lab 5—like distance_km and estimated_delivery_days—the F1-score jumped to 0.47 and the ROC-AUC jumped to 0.78. These extra features gave the model clear clues about shipping delays.

7. Would you deploy this model? Why or why not?

Yes, but as an assistant tool for the logistics team, not an automated system that makes decisions alone.

Why: Catching over 72% of late deliveries gives customer support a big advantage. They can reach out to customers early with updates or give the package priority shipping.
Caution: Because Precision is around 35%, about two-thirds of the alerts will be false alarms. So the actions we take must be cheap (like sending a tracking email), not expensive (like issuing an immediate full refund).
4. Challenge Exercise: Changing the Threshold

Normally, if the model is at least 50% sure an order is late, it flags it as late. But we can change this threshold rule depending on what the business cares about:

Threshold	Precision	Recall	F1-Score	What happens in real life
0.30	23.5%	88.4%	0.371	Super sensitive: catches 88% of late orders, but lots of false alarms.
0.50	34.8%	72.5%	0.470	Default balanced setting.
0.70	54.1%	39.2%	0.455	Very cautious: almost no false alarms, but misses over 60% of late orders.
Question: Which threshold would you choose if missing a late order were very expensive?

Choice: 0.30
Reason: If missing a late delivery costs a lot of money or loses valuable customers, we want to avoid False Negatives at all costs. Lowering the threshold to 0.30 tells the model: "If there is even a 30% chance this order is late, warn us immediately." This pushes Recall up to 88.4%, catching almost 9 out of every 10 late packages.

5. Reflection

Before Lab 7:

"I thought a good ML model just meant having a high accuracy number like 95%, and that testing it once was enough to know it worked."

After Lab 7:

"I now understand that high accuracy can be fake when data is imbalanced. A real evaluation means checking the confusion matrix, balancing precision and recall, and using cross-validation to prove the model is reliable."

Most Important Metric:

"For this problem, the most important metric is Recall because failing to warn a customer about a late package destroys trust in our business."

Cross-Validation Lesson:

"Cross-validation is useful because a single test score might just be good luck. Testing the model on 5 different folds gives an honest, trustworthy average."

Deployment Decision:

"I would deploy this model as an early warning alert for our support team so they can check risky packages before customers start complaining."
6. Other In-Lab Questions Answered
Part A: What would happen if a model predicted only the majority class?
It would get a deceptive 92% accuracy, but its Recall for late deliveries would be 0%, making it totally useless.
Part J: Is the minority class being detected well?
Yes. Catching over 72% of a rare class (that only makes up 8% of the data) is a very good result for a baseline model.
Part Q: Why use a Pipeline?
If we clean and scale the data before splitting, information from the test set leaks into the training set (called "data leakage"). A Pipeline keeps the training and testing sets completely separate inside every single fold.