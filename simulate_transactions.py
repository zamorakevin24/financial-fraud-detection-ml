import time
import random
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"


def send_transaction(transaction_id: int, payload: dict):
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        label = result["label"]
        probability = result["fraud_probability"]
        risk = result["risk_level"]

        if label == "fraud":
            print(f"\n[ALERT] Transaction {transaction_id}")
        else:
            print(f"\n[OK] Transaction {transaction_id}")

        print(f"Prediction: {result['fraud_prediction']} ({label})")
        print(f"Fraud probability: {probability:.6f}")
        print(f"Risk level: {risk}")
    else:
        print(f"\n[ERROR] Transaction {transaction_id}")
        print("Request failed:", response.status_code, response.text)


def main():
    df = pd.read_csv("data/raw/creditcard.csv")

    # quitamos Class porque eso no se manda a la API
    normal_df = df[df["Class"] == 0].drop("Class", axis=1)
    fraud_df = df[df["Class"] == 1].drop("Class", axis=1)

    # mezcla de normales y fraudes para simular tráfico
    normal_samples = normal_df.sample(10, random_state=42).to_dict(orient="records")
    fraud_samples = fraud_df.sample(5, random_state=42).to_dict(orient="records")

    transactions = normal_samples + fraud_samples
    random.shuffle(transactions)

    print("Starting transaction simulation...\n")

    for i, tx in enumerate(transactions, start=1):
        send_transaction(i, tx)
        time.sleep(1.5)  # espera entre transacciones


if __name__ == "__main__":
    main()