import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

df = pd.read_csv("data/raw/creditcard.csv")

# 3 normales y 3 fraudes
normal_samples = df[df["Class"] == 0].drop("Class", axis=1).head(3)
fraud_samples = df[df["Class"] == 1].drop("Class", axis=1).head(3)

samples = pd.concat([normal_samples, fraud_samples], ignore_index=True)

for i, row in samples.iterrows():
    payload = row.to_dict()

    response = requests.post(API_URL, json=payload)

    print(f"\nTransaction {i+1}")
    print("Status code:", response.status_code)
    print("Response:", response.json())