# train_transformer_all_asia.py

import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

# === Paths ===
RESULTS_DIR = "results/transformer"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Load Preprocessed Data ===
data = np.load("models/lstm/preprocessed_asia.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

rmse_results = []

# === Transformer Block ===
def transformer_block(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# === Build Transformer Model ===
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(1)(x)
    return Model(inputs, x)

# === Train & Evaluate ===
def train_country(country):
    print(f"\n=== {country} ===")
    X = data[country]["X"]
    y = data[country]["y"]

    # Split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build & train
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer=Adam(0.001), loss="mse")
    history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test, y_test), verbose=0)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")
    rmse_results.append((country, rmse))

    # Plot example
    plt.figure(figsize=(8, 4))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"{country} - Transformer Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_forecast.png")
    plt.close()

# === Loop through countries ===
for country in selected_countries:
    train_country(country)

# === Save RMSE ===
with open(os.path.join(RESULTS_DIR, "transformer_rmse_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE"])
    writer.writerows(rmse_results)

print("\n✅ Done. Results saved to results/transformer/transformer_rmse_results.csv")
