from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data()

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "models/logistic_regression.pkl")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# Neural Network
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

es = EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train, y_train, validation_split=0.2, epochs=20, callbacks=[es])
model.save("models/neural_network.h5")

print("Models trained and saved successfully.")
