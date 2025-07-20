import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import os

WINDOW_SIZE = 30
FORECAST_HORIZON = 7
SELECTED_COUNTRIES = [
    "Vietnam", "Singapore", "Bangladesh",
    "India", "Philippines", "Thailand"
]

def load_data():
    # ✅ Use the fixed CSV file with forward-filled recovered data
    df = pd.read_csv("data/time_series_covid19_recovered_global_fixed.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def apply_wavelet(signal, wavelet="db1", level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # Only keep approximation
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def create_multistep_sequences(series, window_size=WINDOW_SIZE, horizon=FORECAST_HORIZON):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + horizon])
    return np.array(X), np.array(y)

def preprocess_country(df, country_name):
    country_df = df[df["Country"] == country_name].copy()
    raw_cases = country_df["Recovered"].values

    # 🧼 Trim off trailing flat data after last meaningful update
    diffs = np.diff(raw_cases)
    if np.any(diffs != 0):
        last_valid_index = np.max(np.nonzero(diffs)[0]) + 1
        trimmed = raw_cases[:last_valid_index + 1]
    else:
        trimmed = raw_cases  # All values are constant

    # 📉 Smooth the trimmed signal
    smoothed = apply_wavelet(trimmed)

    # 🔢 Normalize
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(smoothed.reshape(-1, 1)).flatten()

    # ⏱ Create sequences for multistep forecasting
    X, y = create_multistep_sequences(normalized)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler

def main():
    df = load_data()
    country_data = {}

    for country in SELECTED_COUNTRIES:
        X, y, scaler = preprocess_country(df, country)
        country_data[country] = {
            "X": X,
            "y": y,
            "scaler": scaler
        }
        print(f"{country}: X shape = {X.shape}, y shape = {y.shape}")

    # ✅ Save to the correct output file
    os.makedirs("models/lstm", exist_ok=True)
    outpath = "models/lstm/recovered_multistep_asia_6countries.npy"
    np.save(outpath, country_data, allow_pickle=True)
    print(f"\n✅ Saved preprocessed data to: {outpath}")

if __name__ == "__main__":
    main()
