import pandas as pd

# Input and output files
INPUT_CSV = "data/countries-aggregated.csv"
OUTPUT_CSV = "data/time_series_covid19_recovered_global_fixed.csv"

# Countries to fix
AFFECTED_COUNTRIES = ["Vietnam", "Singapore", "Bangladesh", "India", "Philippines", "Thailand"]

# Load and sort data
df = pd.read_csv(INPUT_CSV, parse_dates=["Date"])
df = df.sort_values(["Country", "Date"])

# Store fixed data
fixed_data = []

for country, group in df.groupby("Country"):
    group = group.copy()
    group.reset_index(drop=True, inplace=True)

    if country in AFFECTED_COUNTRIES:
        recovered = group["Recovered"]

        # Step 1: Detect when recovered drops from non-zero to 0
        drop_index = None
        for i in range(1, len(recovered)):
            if recovered[i - 1] > 0 and recovered[i] == 0:
                drop_index = i
                break

        # Step 2: If drop found, forward fill from there using last valid value
        if drop_index is not None:
            last_valid = recovered[drop_index - 1]
            for i in range(drop_index, len(recovered)):
                if recovered[i] == 0:
                    recovered[i] = last_valid
                else:
                    break  # stop if it’s updated again later
            group["Recovered"] = recovered

    fixed_data.append(group)

# Final merge and export
df_fixed = pd.concat(fixed_data).sort_values(["Country", "Date"])
df_fixed.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Fixed CSV saved to {OUTPUT_CSV}")
