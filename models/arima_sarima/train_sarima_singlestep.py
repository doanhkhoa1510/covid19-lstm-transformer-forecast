# models/arima_sarima/train_sarima_singlestep.py

import numpy as np
import os
import csv
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# === Suppress SARIMA warnings ===
warnings.filterwarnings("ignore")

# === Paths ===
DATA_PATH = "models/arima_sarima/recovered_series_cutoff.npy"
RESULT_DIR = "results/arima_sarima/plots_sarima_singlestep"
CSV_PATH = "results/arima_sarima/sarima_rmse_recovered_singlestep.csv"

os.makedirs(RESULT_DIR, exist_ok=True)

# === Load preprocessed data ===
data = np.load(DATA_PATH, allow_pickle=True).item()
selected_countries = list(data.keys())

results = []

def train_sarima_for_country(country):
    print(f"\n📊 Training SARIMA for {country}...")

    series = data[country]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    predictions = []

    for t in range(len(test)):
        try:
            model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=1)[0]
        except Exception as e:
            print(f"❌ {country} at step {t}: {e}")
            forecast = train[-1]
        predictions.append(forecast)
        train = np.append(train, test[t])  # update with actual

    rmse = np.sqrt(mean_squared_error(test, predictions))
    nrmse = rmse / np.mean(test)
    print(f"✅ {country} RMSE: {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # Plot predictions
    plt.figure(figsize=(8, 4))
    plt.plot(test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(f"{country} - SARIMA Forecast (Single-step)")
    plt.xlabel("Time Step")
    plt.ylabel("Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{country}_forecast.png")
    plt.close()

# === Run SARIMA for each country ===
for country in selected_countries:
    train_sarima_for_country(country)

# === Save RMSEs ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE", "NRMSE"])
    writer.writerows(results)

print(f"\n📄 Results saved to: {CSV_PATH}")
print(f"📁 Forecast plots saved in: {RESULT_DIR}")
