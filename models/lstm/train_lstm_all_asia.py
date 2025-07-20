import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# === Paths ===
PLOTS_DIR = "results/lstm/plots_recovered_singlestep"
METRICS_PATH = "results/lstm/lstm_rmse_recovered_singlestep.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

# === Load preprocessed single-step data ===
data = np.load("models/lstm/recovered_single_asia_6countries.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

# === Container to store metrics ===
results = []

def train_and_evaluate(country):
    print(f"\n📊 Training LSTM for {country}...")

    X = data[country]["X"]
    y = data[country]["y"]

    # 80/20 Split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # === Model Architecture ===
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='relu'),
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

    # === Prediction ===
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse = rmse / np.mean(y_test)

    print(f"✅ {country} RMSE: {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # === Plot: Prediction vs Actual ===
    plt.figure(figsize=(8, 4))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f"{country} - Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Cases")
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
    writer.writerow(["Country", "RMSE", "NRMSE"])
    writer.writerows(results)

print(f"\n🏁 All done. Metrics saved to: {METRICS_PATH}")
