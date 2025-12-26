import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data()

models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Neural Network": load_model("models/neural_network.h5")
}

for name, model in models.items():
    if name == "Neural Network":
        y_pred = model.predict(X_test).ravel()
    else:
        y_pred = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
