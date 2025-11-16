# models/arima_sarima/train_sarima_multistep.py

import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# === Configurations ===
DATA_PATH = "models/arima_sarima/recovered_series_cutoff.npy"
RESULT_DIR = "results/arima_sarima/plots_sarima_multistep"
CSV_PATH = os.path.abspath("results/arima_sarima/sarima_multistep_by_step.csv")
N_STEPS = 7  # Forecast horizon

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# === Load data ===
data = np.load(DATA_PATH, allow_pickle=True).item()
selected_countries = list(data.keys())

def train_sarima_multistep(country):
    print(f"\n📊 Training SARIMA (Multi-step) for {country}...")

    series = data[country]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    # To store forecast errors per step
    step_actuals = {s: [] for s in range(1, N_STEPS + 1)}
    step_preds = {s: [] for s in range(1, N_STEPS + 1)}

    num_forecasts = (len(test) - N_STEPS) // N_STEPS
    for i in range(num_forecasts):
        start = i * N_STEPS
        end = start + N_STEPS
        history = np.concatenate([train, test[:start]])
        true_values = test[start:end]

        try:
            model = SARIMAX(
                history,
                order=(3, 1, 2),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=N_STEPS)
        except Exception as e:
            print(f"❌ {country} step {i}: {e}")
            forecast = [history[-1]] * N_STEPS

        # Collect per-step predictions
        for s in range(1, N_STEPS + 1):
            if len(true_values) >= s:
                step_actuals[s].append(true_values[s - 1])
                step_preds[s].append(forecast[s - 1])

    # === Compute RMSE/NRMSE per step ===
    step_rmse = []
    step_nrmse = []
    for s in range(1, N_STEPS + 1):
        rmse = np.sqrt(mean_squared_error(step_actuals[s], step_preds[s]))
        nrmse = rmse / np.mean(step_actuals[s])
        step_rmse.append(rmse)
        step_nrmse.append(nrmse)

    print(f"✅ {country} per-step NRMSE: {[round(n, 4) for n in step_nrmse]}")

    # === Plot degradation chart ===
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, N_STEPS + 1), step_nrmse, marker='o', label='NRMSE')
    plt.title(f"{country} - SARIMA Multi-step Performance")
    plt.xlabel("Forecast Horizon (days ahead)")
    plt.ylabel("NRMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{country}_nrmse_by_step.png")
    plt.close()

    return (country, step_rmse, step_nrmse)

# === Run all countries ===
all_results = []
for country in selected_countries:
    res = train_sarima_multistep(country)
    all_results.append(res)

# === Save CSV ===
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
print(f"📁 Plots saved in: {RESULT_DIR}")
