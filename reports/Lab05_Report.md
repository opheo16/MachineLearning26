# Laboratory Experiment 5: Implementation of Ordinary Least Squares (OLS) and Gradient Descent for Linear Regression

**Course:** Machine Learning  
**Unit:** Unit 2 – Supervised Learning & Optimization Foundations  
**Lab Number:** 5  
**Lab Title:** Implementation of Ordinary Least Squares (OLS) and Gradient Descent for Linear Regression  
**Instructor:** Sharad Laad  
**Student Name:** Priyanshi Mali


---

## 1. Executive Summary & Objective

In this laboratory experiment, we implemented simple bivariate Linear Regression ($y = mx + c$) from scratch in Python to model the relationship between **Height** ($X$, independent variable) and **Weight** ($y$, dependent variable). 

We compared two fundamental mathematical paradigms:
1. **The Analytical Approach:** Ordinary Least Squares (OLS) closed-form matrix/algebraic solution.
2. **The Iterative Optimization Approach:** Batch Gradient Descent using Mean Squared Error (MSE) loss minimization.

Furthermore, we computed statistical significance metrics ($R^2$, Standard Error of the slope, Student's $t$-statistic, and two-tailed $p$-value) to evaluate the predictive quality and statistical validity of the fitted parameters.

* **Experimental Dataset:**
  * Sample size: $n = 3$ observations
  * Height ($X$): `[0.5, 2.3, 2.9]`
  * Weight ($y$): `[1.4, 1.9, 3.2]`

---

## 2. Mathematical Foundations

### 2.1 Method 1: Ordinary Least Squares (Closed-Form Analytical Solution)
The closed-form analytical solution minimizes the sum of squared vertical residuals directly without iterative steps:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \quad \bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$$

$$\text{Slope } (m) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

$$\text{Intercept } (c) = \bar{y} - m\bar{x}$$

---

### 2.2 Method 2: Batch Gradient Descent (Iterative Optimization)
Gradient Descent starts with initial guesses ($m_0 = 0.0, c_0 = 0.0$) and iteratively steps in the opposite direction of the gradient of the cost function:

* **Cost Function (Mean Squared Error - MSE):**
  $$J(m, c) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{n} \sum_{i=1}^n (mx_i + c - y_i)^2$$

* **Partial Derivatives (Gradients):**
  $$\frac{\partial J}{\partial m} = \frac{2}{n} \sum_{i=1}^n (mx_i + c - y_i) \cdot x_i$$
  $$\frac{\partial J}{\partial c} = \frac{2}{n} \sum_{i=1}^n (mx_i + c - y_i)$$

* **Parameter Update Rules:**
  $$m := m - \alpha \cdot \frac{\partial J}{\partial m}$$
  $$c := c - \alpha \cdot \frac{\partial J}{\partial c}$$
  *(where $\alpha = 0.05$ is the learning rate, trained across 1,500 epochs)*

---

### 2.3 Statistical Evaluation Metrics
* **Total Sum of Squares ($SS_{\text{tot}}$):** $\sum (y_i - \bar{y})^2 = 1.7267$
* **Residual Sum of Squares ($SS_{\text{res}}$):** $\sum (y_i - \hat{y}_i)^2 = 0.4446$
* **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$
* **Residual Variance ($s_e^2$):** $s_e^2 = \frac{SS_{\text{res}}}{n - 2}$ ($df = 3 - 2 = 1$)
* **Standard Error of Slope ($SE(m)$):** $SE(m) = \sqrt{\frac{s_e^2}{\sum (x_i - \bar{x})^2}}$
* **$t$-Statistic:** $t = \frac{m - 0}{SE(m)}$
* **Two-Tailed $p$-Value:** $p = 2 \cdot (1 - F(|t|, df))$

---

## 3. Experimental Results & Verification

Both models were executed and converged to identical regression parameters:

| Metric / Parameter | Analytical OLS | Batch Gradient Descent | Relative Difference |
| :--- | :---: | :---: | :---: |
| **Slope ($m$)** | **0.6410** | **0.6410** | $< 0.0001$ |
| **Intercept ($c$)** | **0.9487** | **0.9487** | $< 0.0001$ |
| **Fitted Equation** | $y = 0.6410x + 0.9487$ | $y = 0.6410x + 0.9487$ | Identical line |
| **Coefficient of Determination ($R^2$)**| **0.7425** | **0.7425** | Identical |
| **$t$-Statistic** | **1.6984** | **1.6984** | Identical |
| **$p$-Value** | **0.3381** | **0.3381** | Identical |
| **Convergence Epochs** | $1$ (direct calculation) | $1,500$ iterations | N/A |

---

## 4. Visualization & Convergence Analysis

The generated 2-panel plot illustrates two core findings:
1. **Subplot 1 (Regression Fit Comparison):** The fitted OLS line (blue solid) and Gradient Descent line (red dashed) lie precisely on top of one another across the domain $X \in [0, 3.5]$, proving that Gradient Descent successfully reached the global analytical minimum.
2. **Subplot 2 (Loss Convergence Curve):** The MSE loss began high at epoch 0 ($J \approx 5.3$) when $m=0, c=0$, dropped precipitously during the first 200 epochs, and leveled off asymptotically into a flat minimum ($J \approx 0.148$) by epoch 1,000–1,500.

---

## 5. Lab Report & Viva Answers (Section 6 of Manual)

### Q1: Analytical vs. Iterative Comparison
* **Question:** *Do the values of $(m, c)$ and $R^2$ obtained via Gradient Descent match those from OLS? If there is a slight difference, why?*
* **Answer:** Yes, the values match to four decimal places. OLS computes the exact mathematical optimum in one step using calculus. Gradient Descent is an iterative numerical approximation; any tiny difference occurs because Gradient Descent takes finite steps ($\alpha = 0.05$) and approaches the theoretical minimum asymptotically without reaching infinite precision.

---

### Q2: Interpreting $R^2$ vs. $p$-value for Small Sample Sizes ($n = 3$)
* **Question:** *Notice that the $R^2$ score is high ($> 0.74$), indicating strong correlation, yet the $p$-value for the slope exceeds the standard significance threshold ($\alpha = 0.05$). Explain why a high $R^2$ can coexist with a non-significant $p$-value when sample size $n$ is very small ($df = 1$).*
* **Answer:** 
  * $R^2$ measures **effect size** (the proportion of variance in Weight explained by Height). An $R^2 = 0.7425$ indicates that the linear trend is strong in this sample.
  * The $p$-value measures **statistical significance** (the probability of observing this relationship purely by random chance if no real relationship exists). 
  * Because $n = 3$, the degrees of freedom are $df = n - 2 = 1$. With only 1 degree of freedom, statistical power is extremely low, and the critical $t$-value threshold for $5\%$ significance is huge ($t_{\text{crit}} = 12.71$). Our observed $t = 1.6984$ is well below this threshold, yielding $p = 0.3381$. Therefore, we cannot statistically rule out coincidence despite the strong sample fit.

---

### Q3: Hypothesis Testing on the Slope ($m$)
* **Question:** *State the null hypothesis ($H_0$) and alternative hypothesis ($H_1$) tested by the $t$-statistic on the slope $m$.*
* **Answer:**
  * **Null Hypothesis ($H_0$):** $m = 0$ (There is no linear relationship between Height and Weight; Height has zero predictive effect on Weight).
  * **Alternative Hypothesis ($H_1$):** $m \neq 0$ (There is a statistically significant linear relationship between Height and Weight).

---

### Q4: Learning Rate Sensitivity
* **Question:** *What happens to parameter convergence if you increase the learning rate to $\alpha = 1.0$? What happens if you decrease it to $\alpha = 0.0001$?*
* **Answer:**
  * **When $\alpha = 1.0$ (Overshooting / Divergence):** The step size is excessively large. The parameter updates leap across the valley of the cost curve and land higher on the opposite side, causing the loss to oscillate wildly and diverge to infinity (`NaN` / overflow).
  * **When $\alpha = 0.0001$ (Underfitting / Slow Convergence):** The step size is microscopic. The parameters make negligible progress per epoch. After 1,500 iterations, the model remains trapped near the origin ($m \approx 0, c \approx 0$), requiring hundreds of thousands of epochs to converge.

---

### Q5: Computational Scalability
* **Question:** *The closed-form OLS formulation requires computing $(X^T X)^{-1}$. Explain why Gradient Descent is preferred when training models on high-dimensional datasets with millions of features/samples.*
* **Answer:**
  * Inverting the matrix $(X^T X)$ has a computational complexity of $\mathcal{O}(d^3)$, where $d$ is the number of features, and requires storing an entire $d \times d$ matrix in RAM. For high-dimensional data (e.g., $d = 100,000$), exact matrix inversion is computationally impossible.
  * **Gradient Descent** scales linearly $\mathcal{O}(n \cdot d)$ per iteration and only requires basic vector-matrix dot products. When paired with mini-batches or stochastic sampling, Gradient Descent processes arbitrary dataset sizes out-of-core with fixed memory overhead, making it the bedrock optimization algorithm for large-scale Machine Learning and Deep Learning.

---

## 6. Conclusion

In Laboratory Experiment 5, we validated that:
1. Both analytical calculus (OLS) and numerical optimization (Gradient Descent) converge to the exact same optimal decision boundary ($y = 0.6410x + 0.9487$).
2. Gradient Descent operates by systematically following negative error gradients to minimize Mean Squared Error.
3. Statistical metrics like $R^2$ and $p$-value measure complementary properties: effect magnitude versus statistical confidence.
4. While OLS is ideal for small, low-dimensional problems, Gradient Descent provides the mathematical scalability required for modern artificial intelligence.