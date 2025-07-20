# transformer/train_transformer_all_asia_multistep.py

import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

# === Paths ===
PLOTS_DIR = "results/transformer/plots_recovered_multistep"
CSV_PATH = "results/transformer/transformer_rmse_recovered_multistep.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# === Load preprocessed multistep data ===
data = np.load("models/lstm/recovered_multistep_asia_6countries.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

results = []

def transformer_block(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(inputs.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_dim)(x)
    return Model(inputs, outputs)

def train_and_evaluate(country):
    print(f"\n📊 Training Transformer for {country}...")

    X = data[country]["X"]
    y = data[country]["y"]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = build_model(input_shape=(X.shape[1], X.shape[2]), output_dim=y.shape[1])
    model.compile(optimizer=Adam(0.001), loss="mse")

    history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test, y_test), verbose=0)

    # === Loss Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{country} - Loss Curve (Recovered, 7-step)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_loss.png")
    plt.close()

    # === Prediction Plot ===
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ {country} RMSE (7-step): {rmse:.4f}")
    results.append((country, rmse))

    plt.figure(figsize=(8, 4))
    plt.plot(y_test[0], label="Actual (1st sample)")
    plt.plot(y_pred[0], label="Predicted (1st sample)")
    plt.title(f"{country} - 7-Day Forecast (Recovered)")
    plt.xlabel("Day Ahead")
    plt.ylabel("Normalized Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{country}_forecast_sample.png")
    plt.close()

# === Train all countries ===
for country in selected_countries:
    train_and_evaluate(country)

# === Save RMSE to CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "7-Day RMSE (Recovered)"])
    writer.writerows(results)

print(f"\n📁 All plots saved in: {PLOTS_DIR}")
print(f"📄 RMSE results saved to: {CSV_PATH}")
