import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------
companies = {
    "Tata Motors": "TATAMOTORS.NS",
    "Reliance": "RELIANCE.NS",
    "Amazon": "AMZN"
}
window = 60
user_date_str = "2025-09-20"

# -------------------------------
# FEATURE ENGINEERING FUNCTION
# -------------------------------
def add_features(df):
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    return df

# -------------------------------
# DATA PREPARATION
# -------------------------------
def prepare_data(df, window=60):
    features = ["Open", "High", "Low", "Close", "Volume", "MA_10", "MA_50", "EMA_20", "Return"]
    data = df[features].values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i, 3])  # "Close" is index 3

    return np.array(X), np.array(y), scaler

# -------------------------------
# MODEL DEFINITIONS
# -------------------------------
def build_rnn(input_shape):
    model = Sequential([
        SimpleRNN(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        SimpleRNN(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_lstm(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_next_day(model, data, scaler, window=60):
    last_window = data[-window:]
    last_window = np.expand_dims(last_window, axis=0)
    pred_scaled = model.predict(last_window)
    pad = np.zeros((1, data.shape[1]))
    pad[0, 3] = pred_scaled  # Close is index 3
    pred = scaler.inverse_transform(pad)[0, 3]
    return pred

# -------------------------------
# MAIN LOOP
# -------------------------------
for company, ticker in companies.items():
    print(f"\n🔹 Processing {company} ({ticker})...")
    df = yf.download(ticker, start="2015-01-01", end="2025-09-15")
    df = add_features(df)

    X, y, scaler = prepare_data(df, window)
    input_shape = (X.shape[1], X.shape[2])

    models = {
        "RNN": build_rnn(input_shape),
        "LSTM": build_lstm(input_shape),
        "CNN": build_cnn(input_shape),
        "CNN+LSTM": build_cnn_lstm(input_shape)
    }

    predictions = {}
    results = {}

    for name, model in models.items():
        print(f"Training {name} model for {company}...")

        epochs = 100 if name in ["LSTM", "CNN+LSTM"] else 60
        early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

        history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

        # Prediction
        pred_price = predict_next_day(model, X[-window:], scaler)
        predictions[name] = pred_price

        # Evaluation on training
        y_pred = model.predict(X)
        y_true = y
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        results[name] = {"RMSE": rmse, "MAPE": mape}
        print(f"{name} → Predicted: ₹{pred_price:.2f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

    print(f"\n✅ Final predictions for {company} on {user_date_str}:")
    for name, price in predictions.items():
        print(f"{name}: ₹{price:.2f}")
