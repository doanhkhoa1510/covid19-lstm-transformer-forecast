# models/lstm/train_lstm_all_asia_multistep.py

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
RESULTS_DIR = "results/lstm"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_recovered_multistep")
CSV_PATH = os.path.join(RESULTS_DIR, "lstm_rmse_recovered_multistep.csv")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load preprocessed multi-step recovered data ===
data = np.load("models/lstm/recovered_multistep_asia_6countries.npy", allow_pickle=True).item()
selected_countries = list(data.keys())
N_STEPS = 7

def train_and_evaluate(country):
    print(f"\n📊 Training LSTM (7-step) for {country}...")

    X = data[country]["X"]
    y = data[country]["y"]

    # === Train-test split ===
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # === Model Architecture ===
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='tanh'),
        Dense(N_STEPS)
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
    plt.title(f"{country} - Loss Curve (Recovered, 7-step)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_loss_recovered_multistep.png"))
    plt.close()

    # === Prediction ===
    y_pred = model.predict(X_test)

    # === Compute per-step RMSE & NRMSE ===
    step_rmse, step_nrmse = [], []
    for s in range(N_STEPS):
        rmse = np.sqrt(mean_squared_error(y_test[:, s], y_pred[:, s]))
        nrmse = rmse / np.mean(y_test[:, s])
        step_rmse.append(rmse)
        step_nrmse.append(nrmse)

    print(f"✅ {country} per-step NRMSE: {[round(n, 4) for n in step_nrmse]}")

    # === Plot NRMSE degradation ===
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, N_STEPS + 1), step_nrmse, marker='o', label='NRMSE')
    plt.title(f"{country} - Multi-step Performance (LSTM)")
    plt.xlabel("Forecast Horizon (days ahead)")
    plt.ylabel("NRMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_nrmse_by_step.png"))
    plt.close()

    return (country, step_rmse, step_nrmse)

# === Train across all countries ===
all_results = []
for country in selected_countries:
    res = train_and_evaluate(country)
    all_results.append(res)

# === Save results to CSV ===
header = (
    ["Country"]
    + [f"Step_{i}_RMSE" for i in range(1, N_STEPS + 1)]
    + [f"Step_{i}_NRMSE" for i in range(1, N_STEPS + 1)]
)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for country, rmses, nrs in all_results:
        writer.writerow([country] + rmses + nrs)

print(f"\n📄 Per-step results saved to: {CSV_PATH}")
print(f"📁 Plots saved in: {PLOTS_DIR}")
