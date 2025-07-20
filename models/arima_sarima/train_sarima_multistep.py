import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

# === Suppress warnings for clean output ===
warnings.filterwarnings("ignore")

# === Configurations ===
DATA_PATH = "models/arima_sarima/recovered_series_cutoff.npy"
RESULT_DIR = "results/arima_sarima/plots_sarima_multistep"
CSV_PATH = "results/arima_sarima/sarima_rmse_recovered_multistep.csv"
N_STEPS = 7  # Forecast horizon

os.makedirs(RESULT_DIR, exist_ok=True)

# === Load data ===
data = np.load(DATA_PATH, allow_pickle=True).item()
selected_countries = list(data.keys())
results = []

def train_sarima_multistep(country):
    print(f"\n📊 Training SARIMA (Multi-step) for {country}...")

    series = data[country]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    predictions = []
    actuals = []

    for i in range(0, len(test) - N_STEPS + 1):
        history = np.concatenate([train, test[:i]])
        true_values = test[i : i + N_STEPS]

        try:
            model = SARIMAX(history, order=(3, 1, 2), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=N_STEPS)
        except Exception as e:
            print(f"❌ {country} at step {i}: {e}")
            forecast = [history[-1]] * N_STEPS  # fallback

        predictions.extend(forecast)
        actuals.extend(true_values)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    nrmse = rmse / np.mean(actuals)
    print(f"✅ {country} RMSE: {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(actuals, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(f"{country} - SARIMA Forecast (Multi-step)")
    plt.xlabel("Time Step")
    plt.ylabel("Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{country}_forecast.png")
    plt.close()

# === Train for all countries ===
for country in selected_countries:
    train_sarima_multistep(country)

# === Save RMSE results to CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE", "NRMSE"])
    writer.writerows(results)

print(f"\n📄 Results saved to: {CSV_PATH}")
print(f"📁 Forecast plots saved in: {RESULT_DIR}")
