# Financial Fraud Detection using Hybrid Machine Learning Models

This project implements a complete fraud detection system using a hybrid machine learning architecture that combines anomaly detection and supervised learning models. The system detects potentially fraudulent credit card transactions and exposes the model through an API for real-time inference.

---

# Project Overview

Financial fraud detection is a critical task in modern banking systems due to the large number of daily transactions and the high cost associated with fraudulent activity.

This project explores multiple machine learning approaches for detecting fraudulent transactions using the well-known **Credit Card Fraud Detection dataset**. The goal is to evaluate different models and design a hybrid architecture that improves fraud detection while maintaining low false positive rates.

The final system includes:

- Exploratory Data Analysis (EDA)
- Multiple machine learning models
- A hybrid anomaly detection + classification pipeline
- A real-time prediction API
- A transaction simulation script

---

# Dataset

Dataset used:

Credit Card Fraud Detection Dataset  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- Highly imbalanced dataset
- Features transformed using PCA (V1–V28)
- Additional features:
  - `Time`
  - `Amount`
  - `Class` (target variable)

Class distribution:

- Normal transactions: ~99.83%
- Fraud transactions: ~0.17%

---

# Machine Learning Models Evaluated

Several models were tested and compared:

### Logistic Regression
Baseline model for comparison.

### Random Forest
Tree-based ensemble model capable of capturing nonlinear patterns.

### XGBoost
Gradient boosting model known for strong performance on tabular data.

### Isolation Forest
Unsupervised anomaly detection model used to identify unusual transaction behavior.

### Hybrid Models
A hybrid approach was implemented where:

1. Isolation Forest generates an **anomaly score**
2. The anomaly score is used as an additional feature
3. A supervised model (Random Forest or XGBoost) performs the final classification

---

# Hybrid Model Architecture

The final system uses a pipeline architecture:

Transaction
↓
Feature Engineering
↓
Isolation Forest (Anomaly Detection)
↓
Anomaly Score
↓
XGBoost Classifier
↓
Fraud Probability


This approach allows the model to combine:

- behavioral anomaly detection
- learned fraud patterns

---

# Project Structure
financial_fraud_detection_ml/

data/
raw/creditcard.csv

models/
hybrid_pipeline.pkl

notebooks/
01_eda.ipynb
02_modeling_baseline.ipynb
03_random_forest.ipynb
04_isolation_forest.ipynb
05_xgboost_model.ipynb
06_hybrid_model.ipynb
07_hybrid_xgboost.ipynb
08_model_comparison.ipynb
09_hybrid_pipeline.ipynb

src/
hybrid_pipeline.py

api/
main.py
schemas.py

simulate_transactions.py
test_api.py

README.md
requirements.txt


---

# Model Performance

Example performance results:

| Model | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|
| Logistic Regression | ~0.79 | ~0.69 | ~0.74 | - |
| Random Forest | ~0.96 | ~0.76 | ~0.85 | ~0.95 |
| Random Forest (threshold tuned) | ~0.94 | ~0.83 | ~0.88 | ~0.95 |
| XGBoost | ~0.81 | ~0.85 | ~0.83 | ~0.97 |
| Hybrid Random Forest | ~0.94 | ~0.82 | ~0.87 | ~0.96 |
| Hybrid XGBoost | ~0.74 | ~0.83 | ~0.78 | ~0.97 |

The hybrid approach allows the classifier to incorporate anomaly detection signals, improving detection capability for unusual transaction patterns.

---

# API for Real-Time Fraud Detection

The trained model is deployed using **FastAPI**.

Endpoint:

Example request:

{
"Time": 0,
"V1": -1.359807,
"V2": -0.072781,
...
"V28": -0.021053,
"Amount": 149.62
}

Example response:

{
"fraud_prediction": 1,
"label": "fraud",
"fraud_probability": 0.9998,
"risk_level": "high"
}

# Running the Project
1 Install dependencies
pip install -r requirements.txt

2 Start the API
uvicorn api.main:app --reload

API documentation will be available at:

http://127.0.0.1:8000/docs

3 Simulate Transactions
To simulate a stream of transactions:

python simulate_transactions.py

Example output:

[ALERT] Transaction 7
Prediction: 1 (fraud)
Fraud probability: 0.999896
Risk level: high

# Technologies Used

Python

Pandas

Scikit-learn

XGBoost

FastAPI

Uvicorn

Jupyter Notebook

Future Improvements

Possible extensions include:

real-time streaming using Kafka

model monitoring and logging

fraud investigation dashboard

model retraining pipeline

# Author

Kevin Zamora

Machine Learning and Data Science Enthusiast