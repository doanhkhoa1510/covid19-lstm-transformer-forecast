import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load RMSE data ===
lstm_df = pd.read_csv("results/lstm/lstm_rmse_results.csv")
trans_df = pd.read_csv("results/transformer/transformer_rmse_results.csv")

# === Align and merge ===
merged = lstm_df.merge(trans_df, on="Country", suffixes=("_LSTM", "_Transformer"))

# === Determine which model performed better per country ===
merged["Better Model"] = merged.apply(
    lambda row: "LSTM" if row["RMSE_LSTM"] < row["RMSE_Transformer"] else "Transformer",
    axis=1
)

# === Save table to CSV ===
comparison_table_path = "results/comparison_single_step_table.csv"
merged.to_csv(comparison_table_path, index=False)
print(f"✅ Saved table to: {comparison_table_path}")

# === Plot grouped bar chart ===
plt.figure(figsize=(12, 6))
bar_width = 0.4
x = range(len(merged))

plt.bar([i - bar_width/2 for i in x], merged["RMSE_LSTM"], width=bar_width, label="LSTM", color='steelblue')
plt.bar([i + bar_width/2 for i in x], merged["RMSE_Transformer"], width=bar_width, label="Transformer", color='orange')

plt.xticks(ticks=x, labels=merged["Country"], rotation=45)
plt.ylabel("RMSE")
plt.title("LSTM vs Transformer RMSE per Country (Single-Step)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()

# === Save chart ===
chart_path = "results/comparison_single_step_chart.png"
plt.savefig(chart_path)
plt.close()

print(f"✅ Saved bar chart to: {chart_path}")
