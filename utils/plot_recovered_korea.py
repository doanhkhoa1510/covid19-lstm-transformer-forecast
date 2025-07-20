import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load dataset ===
df = pd.read_csv("data/countries-aggregated.csv")

# === Filter for South Korea ===
korea_df = df[df["Country"] == "Korea, South"].copy()

# === Convert Date to datetime format ===
korea_df["Date"] = pd.to_datetime(korea_df["Date"])

# === Create line plot ===
plt.figure(figsize=(12, 6))
plt.plot(korea_df["Date"], korea_df["Recovered"], color="green", linewidth=2)

plt.title("Recovered COVID-19 Cases in South Korea", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Recovered Cases")
plt.grid(True)
plt.tight_layout()

# === Save chart to results folder ===
os.makedirs("results", exist_ok=True)
plot_path = "results/recovered_korea.png"
plt.savefig(plot_path)
plt.show()

print(f"✅ Plot saved to {plot_path}")
