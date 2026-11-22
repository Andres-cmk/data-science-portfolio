# 💳 Fraud Detection in Mobile Money Transactions
 
Machine learning system that detects **fraudulent transactions** in a simulated mobile money service, where only **0.13%** of the 6.3 million transactions are fraud. The project follows the **CRISP-DM** methodology and compares a **Logistic Regression** and a **Random Forest** inside scikit-learn pipelines.
 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andres-cmk/data-science-portfolio/blob/main/fraud-detection/Fraud_Detection_System.ipynb)
 
> The notebook's explanations are written in Spanish. This README summarizes the project in English.
 
---
 
## 🎯 Business problem
 
Mobile money lets people move money from their phones without a traditional bank, which is key in regions with limited banking access. Its growth has also increased opportunities for financial fraud. The goal is to build better methods to detect fraudulent transactions automatically.
 
The main challenge: fraud is extremely rare, so a model can look excellent on accuracy while missing most fraud.
 
## 📦 Dataset
 
[Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data) (Kaggle), based on **PaySim**, a financial mobile money simulator built from real transaction logs.
 
| | |
|---|---|
| Transactions | 6,362,620 |
| Columns | 11 |
| Fraudulent | 8,213 (0.13%) |
| Legitimate | 6,354,407 |
 
> ⚠️ The data is **synthetic**, so results may not transfer directly to real-world transactions.
 
## 🔍 Key findings from the analysis
 
- **Fraud only happens in two transaction types:** `TRANSFER` (fraud rate ≈ 0.77%) and `CASH_OUT` (≈ 0.18%). `PAYMENT`, `CASH_IN` and `DEBIT` have no fraud at all.
- **Fraudulent transactions involve larger amounts:** the median fraud amount is about 441,000 versus about 75,000 for legitimate ones.
- **The time variable (`step`) shows no clear relationship with fraud**, so it was dropped.
- **The system's own rule (`isFlaggedFraud`) flags only 16 transactions** out of 6.3 million, so it was excluded from the model.
## 🧠 Approach (CRISP-DM)
 
| Phase | What I did |
|---|---|
| 1. Business understanding | Defined the fraud detection problem for mobile money |
| 2. Data understanding | Explored class imbalance, transaction types, amounts and time |
| 3. Data preparation | Dropped identifiers (`nameOrig`, `nameDest`), `isFlaggedFraud` and `step`; 70/30 train-test split |
| 4. Modeling | Two scikit-learn pipelines with `class_weight="balanced"` to handle the imbalance |
| 5. Evaluation | Compared accuracy, precision, recall and F1, with confusion matrices and feature importance |
| 6. Deployment | Exported both pipelines with `joblib` |
 
**Preprocessing pipeline:** numeric features (`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) are scaled with `StandardScaler`, and `type` is one-hot encoded. Both models share this preprocessing inside a single `Pipeline`, which avoids data leakage between train and test.
 
## 📊 Results
 
Evaluated on **1,908,786 test transactions** (2,435 of them fraudulent):
 
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 94.54% | 2.13% | **93.06%** | 4.17% |
| Random Forest | **99.97%** | **95.71%** | 77.95% | **85.92%** |
 
**How to read it:**
 
- **Accuracy is misleading here.** A model that labels everything as legitimate would already score about 99.9%.
- **Logistic Regression** catches most fraud (recall 93%) but with a precision of only 2%, so almost every alert is a false alarm. In practice, analysts could not review that many cases.
- **Random Forest** gives a much better balance: when it flags a transaction, it is right 96% of the time, and it catches about 78% of all fraud.
<!-- Add your confusion matrices and feature importance chart here, for example:
![Confusion matrix](images/confusion_matrix_rf.png)
![Feature importance](images/feature_importance.png)
-->
 
## ⚠️ Limitations
 
- **Synthetic data:** PaySim simulates transactions, so real-world performance is unknown.
- **Single random split:** the 70/30 split is not stratified and there is no cross-validation, so metrics could vary with another split.
- **No hyperparameter tuning:** the Random Forest uses default settings, and the decision threshold stays at 0.5.
- **About 22% of fraud is still missed** (recall 78%).
## 🔭 Next steps
 
- Use a **stratified split** and **cross-validation**.
- Evaluate with a **precision-recall curve** and tune the decision threshold.
- Try **gradient boosting** (XGBoost, LightGBM) and **resampling** techniques such as SMOTE.
- Engineer features from account balances, such as the difference between expected and actual balance changes.
- Serve the model through an API with **FastAPI**.
## 🚀 How to run
 
1. Click the **Open in Colab** badge above.
2. Download the dataset from Kaggle (link above) and update the CSV path in the notebook. It currently loads the file from Google Drive.
3. Run all cells.
To run it locally:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook Fraud_Detection_System.ipynb
```
 
To reuse the trained models:
 
```python
import joblib
 
model = joblib.load("RandomF.joblib")
model.predict(X_new)   # X_new needs the same columns used for training
```
 
## 🛠️ Tech stack
 
Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn · joblib · Google Colab
 
## 📚 References
 
- Kaggle, *Fraud Detection Dataset*.
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. *PaySim: A financial mobile money simulator for fraud detection.*
## 👤 Author
 
**Andrés Ramírez** · Systems and Computer Engineering student at Universidad Nacional de Colombia
[GitHub](https://github.com/Andres-cmk) · [LinkedIn](https://www.linkedin.com/in/andres-felipe-ramirez-m/)
