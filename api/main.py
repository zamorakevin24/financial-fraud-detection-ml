from fastapi import FastAPI
import pandas as pd
import joblib

from api.schemas import Transaction

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/hybrid_pipeline.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    df = pd.DataFrame([data])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    if probability >= 0.8:
        risk_level = "high"
    elif probability >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    label = "fraud" if prediction == 1 else "normal"

    return {
        "fraud_prediction": prediction,
        "label": label,
        "fraud_probability": probability,
        "risk_level": risk_level
    }