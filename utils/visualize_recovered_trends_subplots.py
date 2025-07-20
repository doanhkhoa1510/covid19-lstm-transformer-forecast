import pandas as pd
import matplotlib.pyplot as plt

SELECTED_COUNTRIES = [
    "Vietnam", "Singapore", "Bangladesh",
    "India", "Philippines", "Thailand"
]

# Load dataset
df = pd.read_csv("data/countries-aggregated.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Create subplots: 2 columns x 3 rows
fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()  # flatten 2D array of axes to 1D

for i, country in enumerate(SELECTED_COUNTRIES):
    country_df = df[df["Country"] == country]
    axs[i].plot(country_df["Date"], country_df["Recovered"], color="teal")
    axs[i].set_title(country)
    axs[i].set_ylabel("Recovered")
    axs[i].grid(True)

# Set a common X label
fig.text(0.5, 0.04, 'Date', ha='center')
fig.suptitle("Recovered COVID-19 Cases Over Time (6 Asian Countries)", fontsize=16)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])  # leave space for suptitle

plt.show()
