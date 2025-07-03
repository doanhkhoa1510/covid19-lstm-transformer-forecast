# train_lstm_vietnam.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load preprocessed Vietnam data
data = np.load("models/lstm/preprocessed_asia.npy", allow_pickle=True).item()
X = data["Vietnam"]["X"]
y = data["Vietnam"]["y"]

# 2. Split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 3. Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss='mse')

# 4. Train
history = model.fit(X_train, y_train, epochs=100, batch_size=8,
                    validation_data=(X_test, y_test), verbose=1)

# 5. Plot training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 6. Predict and Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("LSTM Prediction on Vietnam Test Data")
plt.xlabel("Time Step")
plt.ylabel("Normalized Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
