# models/arima_sarima/train_arima_multistep.py

import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# === Config ===
DATA_PATH = "models/arima_sarima/recovered_series_cutoff.npy"
RESULT_DIR = "results/arima_sarima/plots_arima_multistep"
CSV_PATH = os.path.abspath("results/arima_sarima/arima_rmse_recovered_multistep.csv")
N_STEPS = 7

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# === Load input ===
data = np.load(DATA_PATH, allow_pickle=True).item()
selected_countries = list(data.keys())
results = []

def train_arima_multistep(country):
    print(f"\n📊 Training ARIMA (Multi-step) for {country}...")

    series = data[country]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    actuals = []
    predictions = []

    num_forecasts = (len(test) - N_STEPS) // N_STEPS
    for i in range(num_forecasts):
        start = i * N_STEPS
        end = start + N_STEPS
        history = np.concatenate([train, test[:start]])
        true_values = test[start:end]

        try:
            model = ARIMA(history, order=(3, 1, 2))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=N_STEPS)
        except Exception as e:
            print(f"❌ {country} step {i}: {e}")
            forecast = [history[-1]] * N_STEPS

        actuals.extend(true_values)
        predictions.extend(forecast)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    nrmse = rmse / np.mean(actuals)
    print(f"✅ {country} RMSE: {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # === Save Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(actuals, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(f"{country} - ARIMA Multi-step Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{country}_forecast.png")
    plt.close()

# === Run All Countries ===
for country in selected_countries:
    train_arima_multistep(country)

# === Save CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE", "NRMSE"])
    writer.writerows(results)

print(f"\n📄 Results saved to: {CSV_PATH}")
print(f"📁 Forecast plots saved in: {RESULT_DIR}")
