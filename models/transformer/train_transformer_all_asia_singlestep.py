# models/transformer/train_transformer_all_asia_singlestep.py

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# === Reproducibility ===
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# === Paths ===
PLOTS_DIR = "results/transformer/plots_recovered_singlestep"
METRICS_PATH = "results/transformer/transformer_rmse_recovered_singlestep.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

# === Load preprocessed single-step data (with scalers) ===
data = np.load("models/lstm/recovered_single_asia_6countries.npy", allow_pickle=True).item()
selected_countries = list(data.keys())

# === Transformer Encoder Block ===
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# === Build Transformer Model ===
def build_transformer(input_shape, embed_dim=32, num_heads=2, ff_dim=64, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)  # Single-step output
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# === Container for metrics ===
results = []

def train_and_evaluate(country):
    print(f"\n📊 Training Transformer (single-step) for {country}...")

    X = data[country]["X"]
    y = data[country]["y"]

    # Train-test split (80/20)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build Transformer
    model = build_transformer(X_train.shape[1:])

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Plot Loss Curve
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{country} - Transformer Loss Curve (Single-step)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_loss.png"))
    plt.close()

    # === Predict ===
    y_pred = model.predict(X_test)

    # === Denormalize both predictions and actuals ===
    scaler = data[country]["scaler"]
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # === Compute true RMSE / NRMSE ===
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    nrmse = rmse / np.mean(y_test_denorm)
    print(f"✅ {country} RMSE (unnormalized): {rmse:.4f} | NRMSE: {nrmse:.4f}")
    results.append((country, rmse, nrmse))

    # === Plot Prediction vs Actual (Unnormalized) ===
    plt.figure(figsize=(8, 4))
    plt.plot(y_test_denorm, label="Actual")
    plt.plot(y_pred_denorm, label="Predicted")
    plt.title(f"{country} - Transformer Predictions (Unnormalized)")
    plt.xlabel("Time Step")
    plt.ylabel("Recovered Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_prediction.png"))
    plt.close()

# === Train all countries ===
for country in selected_countries:
    train_and_evaluate(country)

# === Save metrics ===
with open(METRICS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "RMSE (Unnormalized)", "NRMSE"])
    writer.writerows(results)

print(f"\n🏁 All done. Metrics saved to: {METRICS_PATH}")
