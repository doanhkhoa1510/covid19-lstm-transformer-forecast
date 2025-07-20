import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# === Paths ===
RESULTS_DIR = "results/lstm"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_recovered_multistep")
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Load preprocessed multi-step recovered data ===
data = np.load("models/lstm/recovered_multistep_asia_6countries.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

results = []

def train_and_evaluate(country):
    print(f"\n📊 Training LSTM for {country}...")

    X = data[country]["X"]
    y = data[country]["y"]

    # Split 80/20
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Model: input = (7, 1), output = Dense(7)
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')

    # Train
    history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test, y_test), verbose=0)

    # Loss plot
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

    # Predict
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    nrmse = rmse / np.mean(y_test)
    print(f"✅ {country} RMSE (7-step): {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # Forecast sample plot
    plt.figure(figsize=(8, 4))
    plt.plot(y_test[0], label="Actual (1st sample)")
    plt.plot(y_pred[0], label="Predicted (1st sample)")
    plt.title(f"{country} - Forecast (Recovered, 7-day)")
    plt.xlabel("Day Ahead")
    plt.ylabel("Normalized Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_forecast_recovered_multistep.png"))
    plt.close()

# === Train model for each country ===
for country in selected_countries:
    train_and_evaluate(country)

# === Save results to CSV ===
csv_path = os.path.join(RESULTS_DIR, "lstm_rmse_recovered_multistep.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE", "NRMSE"])
    writer.writerows(results)

print(f"\n📁 All plots saved in: {PLOTS_DIR}")
print(f"📄 RMSE results saved to: {csv_path}")
