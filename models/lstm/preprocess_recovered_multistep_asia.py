# models/lstm/train_lstm_all_asia_singlestep.py

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# === Reproducibility ===
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# === Paths ===
PLOTS_DIR = "results/lstm/plots_recovered_singlestep"
METRICS_PATH = "results/lstm/lstm_rmse_recovered_singlestep.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

# === Load preprocessed single-step data (with scalers) ===
data = np.load("models/lstm/recovered_single_asia_6countries.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

# === Container to store metrics ===
results = []

def train_and_evaluate(country):
    print(f"\n📊 Training LSTM for {country}...")

    X = data[country]["X"]
    y = data[country]["y"]

    # 80/20 Split (preserve temporal order)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # === Model Architecture ===
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # === Train ===
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # === Plot: Loss Curve ===
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{country} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_loss.png"))
    plt.close()

    # === Predict ===
    y_pred = model.predict(X_test)

    # === Denormalize both predictions and actuals ===
    scaler = data[country]["scaler"]
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # === Compute true RMSE / NRMSE ===
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    nrmse = rmse / np.mean(y_test_denorm)
    print(f"✅ {country} RMSE (unnormalized): {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # === Plot: Prediction vs Actual ===
    plt.figure(figsize=(8, 4))
    plt.plot(y_test_denorm, label='Actual')
    plt.plot(y_pred_denorm, label='Predicted')
    plt.title(f"{country} - Actual vs Predicted (Unnormalized)")
    plt.xlabel("Time Step")
    plt.ylabel("Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_prediction.png"))
    plt.close()

# === Train across all countries ===
for country in selected_countries:
    train_and_evaluate(country)

# === Save metrics ===
with open(METRICS_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE (Unnormalized)", "NRMSE"])
    writer.writerows(results)

print(f"\n🏁 All done. Metrics saved to: {METRICS_PATH}")
