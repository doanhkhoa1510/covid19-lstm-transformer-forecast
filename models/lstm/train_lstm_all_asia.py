# train_lstm_all_asia.py

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# === Set up paths ===
RESULTS_DIR = "results/lstm"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Load all country data ===
data = np.load("models/lstm/preprocessed_asia.npy", allow_pickle=True).item()
selected_countries = list(data.keys())  # ['Vietnam', 'Thailand', ..., etc.]

rmse_results = []

def train_and_evaluate(country):
    print(f"\n=== {country} ===")

    X = data[country]["X"]
    y = data[country]["y"]

    # Split 80/20
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')

    # Train
    history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test, y_test), verbose=0)

    # Plot: Loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{country} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_loss.png")
    plt.close()

    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE for {country}: {rmse:.4f}")
    rmse_results.append((country, rmse))

    # Plot: Actual vs Predicted
    plt.figure(figsize=(8, 4))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f"{country} - Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_prediction.png")
    plt.close()

# === Train on all countries ===
for country in selected_countries:
    train_and_evaluate(country)

# === Save RMSE results to CSV ===
csv_path = os.path.join(RESULTS_DIR, "lstm_rmse_results.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE"])
    writer.writerows(rmse_results)

print(f"\n✅ All done. Results saved to: {RESULTS_DIR}")
