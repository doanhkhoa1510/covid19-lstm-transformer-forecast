import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten
from tensorflow.keras.optimizers import Adam

# === Set up paths ===
RESULTS_DIR = "results/transformer"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_multistep")
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Load data ===
data = np.load("models/lstm/preprocessed_asia_multistep.npy", allow_pickle=True).item()
countries = list(data.keys())
rmse_results = []

# === Define Transformer block ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(inputs.shape[-1])(ffn)
    x = Add()([x, ffn])
    return LayerNormalization(epsilon=1e-6)(x)

# === Define model builder ===
def build_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.1)
    x = Flatten()(x)
    outputs = Dense(output_dim)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

# === Train each country ===
for country in countries:
    print(f"\n=== {country} ===")
    X = data[country]["X"]
    y = data[country]["y"]

    # Split 80/20
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = build_model(X_train.shape[1:], output_dim=y_train.shape[1])

    history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test, y_test), verbose=0)

    # Plot loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title(f"{country} - Transformer Loss Curve (7-step)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_loss.png"))
    plt.close()

    # Predict
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    rmse_results.append((country, rmse))
    print(f"7-step RMSE: {rmse:.4f}")

    # Plot forecast sample
    plt.figure(figsize=(8, 4))
    plt.plot(y_test[0], label="Actual (1st sample)")
    plt.plot(y_pred[0], label="Predicted (1st sample)")
    plt.title(f"{country} - 7-Day Forecast Example")
    plt.xlabel("Day Ahead")
    plt.ylabel("Normalized Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_forecast_sample.png"))
    plt.close()

# === Save RMSE to CSV ===
rmse_csv_path = os.path.join(RESULTS_DIR, "transformer_rmse_multistep.csv")
with open(rmse_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "7-Day RMSE"])
    writer.writerows(rmse_results)

print(f"\n✅ Saved RMSE results to: {rmse_csv_path}")
