import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pywt
import numpy as np

def apply_wavelet(signal, wavelet="db1", level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

# === 1. Load data ===
df = pd.read_csv("data/countries-aggregated.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Country"] == "Vietnam") & (df["Date"] <= "2020-03-31")]

raw_series = df["Confirmed"].values
smoothed_series = apply_wavelet(raw_series)

# === 2. Run ADF test ===
result = adfuller(smoothed_series)

print("=== ADF Test on Wavelet-Smoothed Series ===")
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

# === 3. Interpretation ===
if result[1] < 0.05:
    print("✅ Series is stationary (no unit root)")
else:
    print("❌ Series is non-stationary (has unit root)")
