import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("models/random_forest.pkl")

# Load dataset
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features exactly like training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def risk_score(transaction):
    probability = model.predict_proba(transaction.reshape(1, -1))[0][1]
    return int(probability * 100)

# Pick a known fraud transaction
fraud_index = y[y == 1].index[0]
transaction = X_scaled[fraud_index]

print("Fraud Risk Score:", risk_score(transaction))

# Normal transaction
normal_index = y[y == 0].index[0]
normal_tx = X_scaled[normal_index]

print("Normal Transaction Risk:", risk_score(normal_tx))
print("Fraud Transaction Risk:", risk_score(transaction))
