# models/transformer/train_transformer_all_asia_multistep.py

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

# === Paths / Config ===
PLOTS_DIR = "results/transformer/plots_recovered_multistep"
CSV_PATH = "results/transformer/transformer_rmse_recovered_multistep.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

DATA_PATH = "models/lstm/recovered_multistep_asia_6countries.npy"
data = np.load(DATA_PATH, allow_pickle=True).item()
selected_countries = list(data.keys())

N_STEPS = 7  # forecast horizon
EMBED_DIM = 32
NUM_HEADS = 2
FF_DIM = 64
DROPOUT = 0.1
EPOCHS = 100
BATCH_SIZE = 8

# === Transformer encoder block (fixed to accept training=None) ===
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

# === Build Transformer model for multi-step output ===
def build_transformer_multistep(input_shape, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, n_steps=N_STEPS):
    inputs = Input(shape=input_shape)  # (window, channels) e.g. (30,1)
    # project input's last dim to embed_dim so attention can work
    x = Dense(embed_dim)(inputs)  # shape -> (batch, time, embed_dim)
    x = TransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(n_steps)(x)  # multi-step dense
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# === Train & evaluate per country ===
def train_and_evaluate(country):
    print(f"\n📊 Training Transformer (multi-step) for {country}...")

    X = data[country]["X"]  # shape (samples, window, 1)
    y = data[country]["y"]  # shape (samples, N_STEPS)

    # Train-test split (preserve time order)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build model
    model = build_transformer_multistep(X_train.shape[1:])

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Loss curve plot
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{country} - Transformer Loss Curve (7-step)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_loss_recovered_multistep.png"))
    plt.close()

    # Predict on test
    y_pred = model.predict(X_test)

    # Compute per-step metrics
    step_rmse = []
    step_nrmse = []
    for s in range(N_STEPS):
        # ensure not constant mean zero to avoid division by zero
        actual = y_test[:, s]
        pred = y_pred[:, s]
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mean_actual = np.mean(actual) if np.mean(actual) != 0 else 1.0
        nrmse = rmse / mean_actual
        step_rmse.append(rmse)
        step_nrmse.append(nrmse)

    print(f"✅ {country} per-step NRMSE: {[round(n,4) for n in step_nrmse]}")

    # NRMSE vs horizon plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, N_STEPS + 1), step_nrmse, marker="o", label="NRMSE")
    plt.title(f"{country} - Multi-step Performance (Transformer)")
    plt.xlabel("Forecast Horizon (days ahead)")
    plt.ylabel("NRMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{country}_nrmse_by_step.png"))
    plt.close()

    return country, step_rmse, step_nrmse

# === Run for all countries and save CSV ===
all_results = []
for country in selected_countries:
    res = train_and_evaluate(country)
    all_results.append(res)

# Save wide CSV: Step_1_RMSE ... Step_7_RMSE, Step_1_NRMSE ... Step_7_NRMSE
header = ["Country"] + [f"Step_{i}_RMSE" for i in range(1, N_STEPS+1)] + [f"Step_{i}_NRMSE" for i in range(1, N_STEPS+1)]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for country, rmses, nrs in all_results:
        writer.writerow([country] + rmses + nrs)

print(f"\n📄 Per-step results saved to: {CSV_PATH}")
print(f"📁 Plots saved in: {PLOTS_DIR}")
