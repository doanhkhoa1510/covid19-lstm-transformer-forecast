import pandas as pd

# === Load both comparison tables ===
single_df = pd.read_csv("results/comparison_single_step_table.csv")
multi_df = pd.read_csv("results/comparison_multistep_table.csv")

# === Best-performer counts ===
single_counts = single_df["Better Model"].value_counts()
multi_counts = multi_df["Better Model"].value_counts()

print("📊 Best Model (Single-Step):")
print(single_counts)
print("\n📊 Best Model (Multi-Step):")
print(multi_counts)

# === Average RMSEs ===
avg_rmse_single = {
    "LSTM": single_df["RMSE_LSTM"].mean(),
    "Transformer": single_df["RMSE_Transformer"].mean()
}

avg_rmse_multi = {
    "LSTM": multi_df["RMSE_LSTM"].mean(),
    "Transformer": multi_df["RMSE_Transformer"].mean()
}

print("\n📈 Average RMSE (Single-Step):")
for model, val in avg_rmse_single.items():
    print(f"{model}: {val:.4f}")

print("\n📈 Average RMSE (Multi-Step):")
for model, val in avg_rmse_multi.items():
    print(f"{model}: {val:.4f}")

# === Optional: Save summary to file ===
summary_path = "results/model_summary_stats.txt"
with open(summary_path, "w") as f:
    f.write("=== LSTM vs Transformer Summary ===\n\n")

    f.write("Best Model (Single-Step):\n")
    f.write(single_counts.to_string())
    f.write("\n\nBest Model (Multi-Step):\n")
    f.write(multi_counts.to_string())
    f.write("\n\nAverage RMSE (Single-Step):\n")
    for model, val in avg_rmse_single.items():
        f.write(f"{model}: {val:.4f}\n")

    f.write("\nAverage RMSE (Multi-Step):\n")
    for model, val in avg_rmse_multi.items():
        f.write(f"{model}: {val:.4f}\n")

print(f"\n✅ Summary saved to: {summary_path}")
