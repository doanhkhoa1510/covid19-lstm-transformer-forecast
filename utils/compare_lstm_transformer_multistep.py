import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load RMSE data ===
lstm_df = pd.read_csv("results/lstm/lstm_rmse_multistep.csv")
trans_df = pd.read_csv("results/transformer/transformer_rmse_multistep.csv")

# Rename columns to match
lstm_df.columns = ["Country", "RMSE_LSTM"]
trans_df.columns = ["Country", "RMSE_Transformer"]

# === Merge ===
merged = lstm_df.merge(trans_df, on="Country")

# === Determine better model ===
merged["Better Model"] = merged.apply(
    lambda row: "LSTM" if row["RMSE_LSTM"] < row["RMSE_Transformer"] else "Transformer",
    axis=1
)

# === Save table ===
table_path = "results/comparison_multistep_table.csv"
merged.to_csv(table_path, index=False)
print(f"✅ Saved table to: {table_path}")

# === Plot bar chart ===
plt.figure(figsize=(12, 6))
bar_width = 0.4
x = range(len(merged))

plt.bar([i - bar_width/2 for i in x], merged["RMSE_LSTM"], width=bar_width, label="LSTM", color='steelblue')
plt.bar([i + bar_width/2 for i in x], merged["RMSE_Transformer"], width=bar_width, label="Transformer", color='orange')

plt.xticks(ticks=x, labels=merged["Country"], rotation=45)
plt.ylabel("7-Day RMSE")
plt.title("LSTM vs Transformer RMSE per Country (Multi-Step Forecast)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()

# === Save chart ===
chart_path = "results/comparison_multistep_chart.png"
plt.savefig(chart_path)
plt.close()

print(f"✅ Saved bar chart to: {chart_path}")
