import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# === Suppress warnings from statsmodels (convergence, stationarity, etc.) ===
warnings.filterwarnings("ignore")

# === Paths ===
DATA_PATH = "models/arima_sarima/recovered_series_cutoff.npy"
PLOT_DIR = "results/arima_sarima/plots_arima_singlestep"
CSV_PATH = "results/arima_sarima/arima_rmse_recovered_singlestep.csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# === Load preprocessed input ===
data = np.load(DATA_PATH, allow_pickle=True).item()
selected_countries = list(data.keys())

results = []

def train_arima_for_country(country):
    print(f"\n📊 Training ARIMA for {country}...")

    series = data[country]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    predictions = []

    for t in range(len(test)):
        try:
            model = ARIMA(train, order=(3, 1, 2))  # You can tune this
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
        except Exception as e:
            print(f"❌ Error for {country} at step {t}: {e}")
            forecast = train[-1]  # fallback to last known value
        predictions.append(forecast)
        train = np.append(train, test[t])  # update with true value

    rmse = np.sqrt(mean_squared_error(test, predictions))
    nrmse = rmse / np.mean(test)
    print(f"✅ {country} RMSE: {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # Optional: Save prediction plot
    plt.figure(figsize=(8, 4))
    plt.plot(test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(f"{country} - ARIMA Forecast (Single-step)")
    plt.xlabel("Time Step")
    plt.ylabel("Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{country}_forecast.png")
    plt.close()

# === Run for all countries ===
for country in selected_countries:
    train_arima_for_country(country)

# === Save RMSE results ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE", "NRMSE"])
    writer.writerows(results)

print(f"\n📄 Results saved to: {CSV_PATH}")
print(f"📁 Forecast plots saved in: {PLOT_DIR}")
