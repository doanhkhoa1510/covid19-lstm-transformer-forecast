import numpy as np

data = np.load('models/lstm/recovered_multistep_asia_6countries.npy', allow_pickle=True).item()

for country in ['Singapore', 'Thailand']:
    print(f"\n{country}")

    country_data = data[country]
    y = country_data['y']  # shape: (samples, 7)
    scaler = country_data['scaler']

    # Get the last 5 samples (each is a 7-day forecast)
    y_last_scaled = y[-5:]

    # Inverse transform to original scale
    y_last_original = scaler.inverse_transform(y_last_scaled)

    print("Last 5 y (scaled):\n", y_last_scaled)
    print("Last 5 y (original):\n", y_last_original)
