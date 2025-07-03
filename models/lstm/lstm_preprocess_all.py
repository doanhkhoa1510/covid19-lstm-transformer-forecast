# lstm_preprocess_all.py (single step)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import os

WINDOW_SIZE = 7
SELECTED_COUNTRIES = [
    "Vietnam", "Thailand", "India", "Philippines",
    "Indonesia", "Malaysia", "Japan", "Korea, South",
    "Bangladesh", "Pakistan"
]

def load_data():
    df = pd.read_csv("data/countries-aggregated.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= "2020-03-31"]  # Match thesis date range
    return df

def apply_wavelet(signal, wavelet="db1", level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def create_sequences(series, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

def preprocess_country(df, country_name):
    country_df = df[df["Country"] == country_name].copy()
    raw_cases = country_df["Confirmed"].values

    # Wavelet smoothing
    smoothed = apply_wavelet(raw_cases)

    # Normalize
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(smoothed.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_sequences(normalized)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # For LSTM input

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

    # Save to use in training later if needed
    np.save("models/lstm/preprocessed_asia.npy", country_data, allow_pickle=True)

if __name__ == "__main__":
    main()
