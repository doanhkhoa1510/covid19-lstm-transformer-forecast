import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_rmse_bar(csv_path, title, output_path, ylabel="RMSE"):
    df = pd.read_csv(csv_path)
    df = df.sort_values(df.columns[1], ascending=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df.iloc[:, 0], df.iloc[:, 1], color='skyblue')
    plt.title(title)
    plt.xlabel("Country")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Annotate values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved chart: {output_path}")


if __name__ == "__main__":
    # LSTM Single-Step
    plot_rmse_bar(
        csv_path="results/lstm/lstm_rmse_results.csv",
        title="LSTM RMSE (Single-Step Forecast)",
        output_path="results/lstm/lstm_rmse_comparison_chart.png"
    )

    # LSTM Multi-Step
    plot_rmse_bar(
        csv_path="results/lstm/lstm_rmse_multistep.csv",
        title="LSTM 7-Day Forecast RMSE (Multi-Step)",
        output_path="results/lstm/lstm_rmse_multistep_chart.png"
    )

    # Transformer Single-Step
    plot_rmse_bar(
        csv_path="results/transformer/transformer_rmse_results.csv",
        title="Transformer RMSE (Single-Step Forecast)",
        output_path="results/transformer/transformer_rmse_chart.png"
    )

    # Transformer Multi-Step
    plot_rmse_bar(
        csv_path="results/transformer/transformer_rmse_multistep.csv",
        title="Transformer 7-Day Forecast RMSE (Multi-Step)",
        output_path="results/transformer/transformer_rmse_multistep_chart.png"
    )
