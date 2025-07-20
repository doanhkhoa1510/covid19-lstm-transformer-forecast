# utils/prepare_arima_sarima_input.py

import pandas as pd
import numpy as np
import os

# === Settings ===
SOURCE_PATH = "data/time_series_covid19_recovered_global_fixed.csv"
OUTPUT_PATH = "models/arima_sarima/recovered_series_cutoff.npy"
SELECTED_COUNTRIES = [
    "Vietnam", "Singapore", "Bangladesh",
    "India", "Philippines", "Thailand"
]
CUTOFF_DATE = "2021-08-04"

# === Load Data ===
df = pd.read_csv(SOURCE_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# === Prepare Output Dictionary ===
series_dict = {}

for country in SELECTED_COUNTRIES:
    country_df = df[df["Country"] == country].copy()
    country_df = country_df[country_df["Date"] <= CUTOFF_DATE]

    if country_df.empty:
        print(f"⚠️ {country}: No data before cutoff, skipping.")
        continue

    values = country_df["Recovered"].values.astype(float)

    if np.all(values == 0):
        print(f"⚠️ {country}: All zeros in recovered data, skipping.")
        continue

    series_dict[country] = values
    print(f"✅ {country}: {len(values)} entries")

# === Save Dictionary ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, series_dict, allow_pickle=True)
print(f"\n✅ Saved recovered series to: {OUTPUT_PATH}")
