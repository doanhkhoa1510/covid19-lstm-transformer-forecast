import pandas as pd
import matplotlib.pyplot as plt

# Define the countries you're interested in
SELECTED_COUNTRIES = [
    "Vietnam", "Singapore", "Bangladesh",
    "India", "Philippines", "Thailand"
]

# Load the dataset
df = pd.read_csv("data/countries-aggregated.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Create a plot for each country
plt.figure(figsize=(14, 8))

for country in SELECTED_COUNTRIES:
    country_df = df[df["Country"] == country]
    plt.plot(country_df["Date"], country_df["Recovered"], label=country)

# Customize the plot
plt.title("Recovered COVID-19 Cases Over Time in Selected Asian Countries")
plt.xlabel("Date")
plt.ylabel("Recovered Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
