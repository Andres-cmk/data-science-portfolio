# 📉 Customer Churn Prediction for a Telecom Company
 
Machine learning project that predicts which telecom customers are likely to **leave the company** (churn), so the business can act before losing them. It follows the **CRISP-DM** methodology and compares three models: a **Decision Tree**, a **Logistic Regression** and a **Multilayer Perceptron**.
 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andres-cmk/data-science-portfolio/blob/main/customer-churn-prediction/Churn.ipynb)
 
> The notebook's explanations are written in Spanish. This README summarizes the project in English.
 
---
 
## 🎯 Business problem
 
Keeping a customer is cheaper than winning a new one. The goal is to identify customers at risk of churning so the company can offer them a retention action in time.
 
The key trade-off: **missing a customer who leaves costs more than offering a retention incentive to someone who would have stayed**, so the project prioritizes **recall** over accuracy.
 
## 📦 Dataset
 
[Iranian Churn Dataset](https://doi.org/10.24432/C5JW3Z) (UCI Machine Learning Repository), collected from an Iranian telecom company over 12 months.
 
| | |
|---|---|
| Customers | 3,150 |
| Churned | 495 (15.7%) |
| Stayed | 2,655 (84.3%) |
| Missing values | None |
 
All features aggregate the **first 9 months** of activity. The churn label is the customer's status at **month 12**, which leaves a 3-month planning gap.
 
Main features: complaints, seconds of use, frequency of use, frequency of SMS, distinct called numbers, subscription length, age group, charge amount, tariff plan and customer value.
 
## 🔍 Key findings from the analysis
 
- **Complaints are the strongest signal.** Of the 241 customers who complained, 200 left (about 83%), and the association with churn is statistically significant (chi-square test).
- **Age group is also related to churn**, with significant differences across groups.
- **Customers who use the service more** (more distinct numbers called, more seconds of use, more SMS) are less likely to churn.
- **`Status` was excluded because of data leakage.** Churn is 47.3% among inactive customers versus 5.3% among active ones, which means both variables describe almost the same event at different moments, not a genuine predictive relationship.
- **Redundant features were removed:** `Customer Value` correlates 0.92 with `Frequency of SMS`, and `Frequency of use` correlates 0.95 with `Seconds of Use`.
## 🧠 Approach (CRISP-DM)
 
| Phase | What I did |
|---|---|
| 1. Business understanding | Framed churn as a retention problem where recall matters most |
| 2. Data understanding | Histograms, boxplots, outliers (IQR), point-biserial correlations and chi-square tests |
| 3. Data preparation | Removed weak, leaking and redundant features; stratified 80/20 split; log transform and scaling |
| 4. Modeling | Decision Tree, Logistic Regression and MLP |
| 5. Evaluation | Compared accuracy, precision, recall and F1 with confusion matrices |
 
**Final features (5):** `Complains`, `Seconds of Use`, `Frequency of SMS`, `Distinct Called Numbers`, `Age Group`.
 
**Preparation details:** stratified 80/20 split (2,520 training and 630 test customers), `log1p` on the skewed variables (`Seconds of Use`, `Frequency of SMS`) and `StandardScaler` fitted on the training set only.
 
**Models:**
 
| Model | Configuration |
|---|---|
| Decision Tree | `max_depth=4`, `class_weight="balanced"` |
| Logistic Regression | `class_weight="balanced"` |
| Multilayer Perceptron | Hidden layers (16, 8), ReLU, early stopping |
 
## 📊 Results
 
Evaluated on **630 test customers** (99 of them churners):
 
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Decision Tree** | 84.92% | 51.41% | **73.74%** | **60.58%** |
| Logistic Regression | 84.13% | 49.65% | 71.72% | 58.68% |
| Multilayer Perceptron | **90.48%** | **100%** | 39.39% | 56.52% |
 
**How to read it:**
 
- **The MLP has the best accuracy, but it is misleading.** It catches only about 4 out of 10 churners, so 6 out of 10 leave without warning. With 84% of customers staying, accuracy rewards predicting the majority class.
- **The Decision Tree gives the best balance.** It detects about 74% of churners, and roughly half of its alerts are real churners.
- **Logistic Regression performs almost the same as the tree**, which suggests the relationship between these features and churn is largely linear.
- **Recommended model: the Decision Tree.** It balances recall and false alarms, and its rules are easy to explain to the business.
**Logistic Regression coefficients** (standardized features):
 
| Feature | Coefficient | Effect on churn risk |
|---|---|---|
| Complains | +1.13 | Raises it the most |
| Distinct Called Numbers | −0.97 | Lowers it |
| Seconds of Use | −0.45 | Lowers it |
| Frequency of SMS | −0.36 | Lowers it |
| Age Group | −0.01 | Almost no effect |
 
<!-- Add your decision tree plot and confusion matrices here, for example:
![Decision tree](images/decision_tree.png)
![Confusion matrices](images/confusion_matrices.png)
-->
 
## ⚠️ Limitations
 
- **Small test set:** only 99 churners in the test set, so metrics can change a lot with another split.
- **Single split, no cross-validation.**
- **Feature selection used the full dataset:** the decisions about which variables to drop came from exploratory analysis done before the train-test split.
- **Class imbalance handled only with `class_weight`** (the MLP uses no weighting at all).
- **Default decision threshold (0.5)**, with no tuning.
- **The dataset comes from one telecom company in one country**, so results may not generalize.
## 🔭 Next steps
 
- Evaluate with **cross-validation** and a **precision-recall curve**, and tune the decision threshold.
- Try **oversampling** (SMOTE) and **gradient boosting** models.
- Add an **unsupervised clustering analysis** to identify customer segments at risk.
- Estimate the **business cost** of false negatives versus false positives to choose the threshold.
## 🚀 How to run
 
1. Click the **Open in Colab** badge above.
2. Download the dataset from UCI (link above) and update the CSV path in the notebook. It currently loads the file from Google Drive.
3. Run all cells.
To run it locally:
 
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn plotly
jupyter notebook Churn.ipynb
```
 
## 🛠️ Tech stack
 
Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · seaborn · Plotly · Google Colab
 
## 📚 References
 
- Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z
## 👤 Author
 
**Andrés Ramírez** · Systems and Computer Engineering student at Universidad Nacional de Colombia
[GitHub](https://github.com/Andres-cmk) · [LinkedIn](https://www.linkedin.com/in/andres-felipe-ramirez-m/)
 