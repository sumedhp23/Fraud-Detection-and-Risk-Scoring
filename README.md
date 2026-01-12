# Fraud Detection & Risk Scoring System using Machine Learning

## Overview
This project implements an end-to-end machine learning system to assess transaction risk and detect fraudulent activity in financial transactions. The primary goal is to improve fraud detection reliability while handling highly imbalanced data and producing interpretable risk scores suitable for real-world decision-making.

The project focuses on building a practical, production-oriented ML pipeline rather than a purely academic or exploratory approach.

---

## Dataset
The dataset contains over **280,000 financial transactions**, with fraudulent transactions representing a very small minority of the data. This severe class imbalance presents a key challenge, requiring careful preprocessing, evaluation, and metric selection.

**Key challenges addressed:**
- Extreme class imbalance
- Risk of high false negatives
- Feature scaling and normalization
- Model evaluation beyond accuracy

---

## Approach
The solution follows a structured machine learning pipeline:

1. Data cleaning and preprocessing
2. Feature scaling and transformation
3. Handling class imbalance
4. Model training and evaluation
5. Probability-based risk scoring

Multiple models were implemented and compared to balance predictive performance and interpretability.

---

## Models & Evaluation
The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- Neural Network (TensorFlow)

Evaluation focused on metrics appropriate for imbalanced classification:
- ROC-AUC
- Precision
- Recall
- Precision–Recall trade-offs

Model performance was iteratively improved through feature engineering, class weighting, and hyperparameter tuning.

---

## Results
- Improved **ROC-AUC from approximately 0.92 to 0.98**
- Reduced false negatives by approximately **30%**
- Generated transaction-level risk probabilities mapped to a **0–100 risk score**

These results demonstrate a balance between detection accuracy and real-world usability, where minimizing missed fraud cases is critical.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow
- Matplotlib / Seaborn

---

## Repository Structure
```bash
fraud-detection-and-risk-scoring/
│
├── data/
│ ├── raw/ # Original dataset (not tracked if large)
│ └── processed/ # Cleaned and transformed data
│
├── notebooks/
│ ├── exploration.ipynb # EDA and initial experimentation
│
├── src/
│ ├── preprocessing.py # Data cleaning and feature engineering
│ ├── train.py # Model training logic
│ ├── evaluate.py # Model evaluation and metrics
│ └── risk_scoring.py # Risk score generation logic
│
├── models/
│ └── saved_models/ # Trained model artifacts
│
├── requirements.txt
└── README.md
```
---

## How to Run
1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt

3. Run preprocessing:

python src/preprocessing.py

4. Train models:

python src/train.py

5. Evaluate performance:

python src/evaluate.py

6. Generate transaction risk scores:

python src/risk_scoring.py


The risk scoring step converts predicted fraud probabilities into a standardized **0–100 transaction risk score**, making the model output suitable for downstream decision-making and analysis.

---

## Notes
This project emphasizes real-world ML considerations such as evaluation strategy, imbalance handling, and system design. It is intended as a practical demonstration of applied machine learning rather than a theoretical benchmark.

