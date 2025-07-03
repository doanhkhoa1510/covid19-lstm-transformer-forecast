# lstm_preprocess_multistep.py (7 steps)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import os

WINDOW_SIZE = 7
FORECAST_HORIZON = 7
SELECTED_COUNTRIES = [
    "Vietnam", "Thailand", "India", "Philippines",
    "Indonesia", "Malaysia", "Japan", "Korea, South",
    "Bangladesh", "Pakistan"
]

def load_data():
    df = pd.read_csv("data/countries-aggregated.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= "2020-03-31"]
    return df

def apply_wavelet(signal, wavelet="db1", level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def create_sequences(series, input_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON):
    X, y = [], []
    for i in range(len(series) - input_size - forecast_horizon + 1):
        X.append(series[i:i + input_size])
        y.append(series[i + input_size:i + input_size + forecast_horizon])
    return np.array(X), np.array(y)

def preprocess_country(df, country_name):
    country_df = df[df["Country"] == country_name].copy()
    raw_cases = country_df["Confirmed"].values
    smoothed = apply_wavelet(raw_cases)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(smoothed.reshape(-1, 1)).flatten()
    X, y = create_sequences(normalized)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def main():
    df = load_data()
    country_data = {}
    for country in SELECTED_COUNTRIES:
        X, y, scaler = preprocess_country(df, country)
        country_data[country] = {"X": X, "y": y, "scaler": scaler}
        print(f"{country}: X = {X.shape}, y = {y.shape}")
    np.save("models/lstm/preprocessed_asia_multistep.npy", country_data, allow_pickle=True)

if __name__ == "__main__":
    main()
