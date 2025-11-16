# models/lstm/preprocess_recovered_single_asia.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import os

# === Configuration ===
WINDOW_SIZE = 7
SELECTED_COUNTRIES = [
    "Vietnam", "Singapore", "Bangladesh",
    "India", "Philippines", "Thailand"
]

def load_data():
    """
    Load recovered case data and apply fixed cutoff date (2021-08-04)
    to ensure consistency with ARIMA/SARIMA and multistep LSTM datasets.
    """
    df = pd.read_csv("data/time_series_covid19_recovered_global_fixed.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 🔒 Fixed cutoff date
    df = df[df["Date"] <= "2021-08-04"]
    
    return df

def apply_wavelet(signal, wavelet="db1", level=2):
    """
    Apply wavelet smoothing by keeping only approximation coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def create_sequences(series, window_size=WINDOW_SIZE):
    """
    Create sliding window sequences for single-step forecasting.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        window = series[i : i + window_size]
        target = series[i + window_size]
        if np.std(window) == 0:  # Skip constant windows
            continue
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

def preprocess_country(df, country_name):
    """
    Process data for a single country:
    - Extract recovered cases
    - Smooth with wavelet
    - Normalize
    - Create (X, y) sequences
    """
    country_df = df[df["Country"] == country_name].copy()
    print(f"{country_name} - Number of entries after cutoff: {len(country_df)}")

    raw_cases = country_df["Recovered"].values

    if np.all(raw_cases == 0) or np.sum(raw_cases) < 10:
        print(f"⚠️ Skipping {country_name}: Not enough recovered case data.")
        return None, None, None

    smoothed = apply_wavelet(raw_cases)

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(smoothed.reshape(-1, 1)).flatten()

    X, y = create_sequences(normalized)
    if X.size == 0:
        print(f"⚠️ Skipping {country_name}: No valid sequences after preprocessing.")
        return None, None, None

    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def main():
    df = load_data()
    country_data = {}

    for country in SELECTED_COUNTRIES:
        X, y, scaler = preprocess_country(df, country)
        if X is not None:
            country_data[country] = {
                "X": X,
                "y": y,
                "scaler": scaler
            }
            print(f"✅ {country}: X shape = {X.shape}, y shape = {y.shape}")

    if not country_data:
        print("❌ No valid countries to save.")
        return

    os.makedirs("models/lstm", exist_ok=True)
    outpath = "models/lstm/recovered_single_asia_6countries.npy"
    np.save(outpath, country_data, allow_pickle=True)
    print(f"\n✅ Saved preprocessed data to: {outpath}")

if __name__ == "__main__":
    main()
