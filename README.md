# Fraud Detection & Risk Scoring System using Machine Learning

---

## Overview
This project implements an end-to-end fraud detection system using machine learning and deep learning techniques. The system predicts fraudulent transactions and assigns a fraud risk score to support real-world decision-making.

---

## Tech Stack
- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy

---

## Models Used
- Logistic Regression
- Random Forest
- Neural Network

---

## Repository Structure
```bash
fraud-detection/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── neural_network.h5
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── risk_scoring.py
│
├── venv/
│   ├── bin/
│   ├── include/
│   ├── lib/
│   ├── share/
│   └── pyvenv.cfg
│
├── creditcard.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Key Highlights
- Handles highly imbalanced data
- Model optimization and evaluation using ROC-AUC
- Risk scoring instead of binary prediction
- Production-style ML pipeline

---

## Dataset
Credit Card Fraud Detection Dataset (Kaggle)
The dataset is not included due to GitHub size limits.
Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud

---

## Author
Sumedh Patil

