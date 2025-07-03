# train_lstm_all_asia_multistep.py

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# === Paths ===
RESULTS_DIR = "results/lstm"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# === Load preprocessed multi-step data ===
data = np.load("models/lstm/preprocessed_asia_multistep.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

rmse_results = []

def train_and_evaluate(country):
    print(f"\n=== {country} ===")
    X = data[country]["X"]
    y = data[country]["y"]

    # Split 80/20
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Model: input = (7, 1), output = Dense(7)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(y_train.shape[1])  # Output 7 values
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')

    # Train
    history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test, y_test), verbose=0)

    # Loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{country} - Loss Curve (7-step)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_loss.png")
    plt.close()

    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))  # RMSE over all predicted points
    print(f"7-step RMSE for {country}: {rmse:.4f}")
    rmse_results.append((country, rmse))

    # Prediction plot: just 1 sample (optional)
    plt.figure(figsize=(8, 4))
    plt.plot(y_test[0], label="Actual (1st sample)")
    plt.plot(y_pred[0], label="Predicted (1st sample)")
    plt.title(f"{country} - 7-Day Forecast Example")
    plt.xlabel("Day Ahead")
    plt.ylabel("Normalized Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_forecast_sample.png")
    plt.close()

# === Train all countries ===
for country in selected_countries:
    train_and_evaluate(country)

# === Save RMSE to CSV ===
with open(os.path.join(RESULTS_DIR, "lstm_rmse_multistep.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "7-Day RMSE"])
    writer.writerows(rmse_results)



print(f"\n✅ Multi-step results saved to: {RESULTS_DIR}/lstm_rmse_multistep.csv")
