import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === File paths for each model and step ===
csv_paths = {
    "LSTM_Single": "results/lstm/lstm_rmse_recovered_singlestep.csv",
    "LSTM_Multi":  "results/lstm/lstm_rmse_recovered_multistep.csv",
    "ARIMA_Single":       "results/arima_sarima/arima_rmse_recovered_singlestep.csv",
    "ARIMA_Multi":        "results/arima_sarima/arima_rmse_recovered_multistep.csv",
    "SARIMA_Single":      "results/arima_sarima/sarima_rmse_recovered_singlestep.csv",
    "SARIMA_Multi":       "results/arima_sarima/sarima_rmse_recovered_multistep.csv"
}

# === Collect all results into one DataFrame ===
all_results = []

for model_name, path in csv_paths.items():
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        all_results.append({
            "Model": model_name,
            "Country": row["Country"],
            "NRMSE": row["NRMSE"]
        })

df_all = pd.DataFrame(all_results)

# === Question 1: Which model is most accurate (lowest avg NRMSE)? ===
model_avg = df_all.groupby("Model")["NRMSE"].mean().sort_values()
best_model = model_avg.idxmin()

# === Question 2: Which country is easiest to predict (lowest avg NRMSE)? ===
country_avg = df_all.groupby("Country")["NRMSE"].mean().sort_values()
best_country = country_avg.idxmin()

# === Display Results ===
print("\n📊 Average NRMSE by Model:")
print(model_avg.round(4))

print("\n🏆 Most accurate model overall:", best_model)

print("\n🌍 Average NRMSE by Country:")
print(country_avg.round(4))

print("\n🏅 Country with best forecast accuracy:", best_country)

# === Visualizations ===
sns.set(style="whitegrid")

# Bar chart: NRMSE by Model
plt.figure(figsize=(10, 6))
sns.barplot(x=model_avg.index, y=model_avg.values, palette="viridis")
plt.title("Average NRMSE by Model")
plt.ylabel("NRMSE")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/avg_nrmse_by_model.png")
plt.show()

# Bar chart: NRMSE by Country
plt.figure(figsize=(10, 6))
sns.barplot(x=country_avg.index, y=country_avg.values, palette="crest")
plt.title("Average NRMSE by Country")
plt.ylabel("NRMSE")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/avg_nrmse_by_country.png")
plt.show()
