"""
model_fitting.py

Functions for fitting univariate and multivariate time-series forecasting models.
Supports:
- Linear Regression
- Random Forest
- XGBoost
- LSTM (Keras)
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ============================================================
# DATA SPLITTING HELPERS
# ============================================================

def split_X_y(df, target="SO2"):
    """
    Splits a dataframe into X (features) and y (target)
    """
    X = df.drop(columns=[target])
    
    # Remove any datetime columns — regardless of frequency
    datetime_cols = X.select_dtypes(include=["datetime", "datetime64"]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=datetime_cols)
    
    y = df[target]
    return X, y


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_linear_regression(**params):
    return LinearRegression(**params)


def build_random_forest(n_estimators=300, max_depth=None):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )


def build_xgb(n_estimators=300, learning_rate=0.05, max_depth=6,
              subsample=0.8, colsample_bytree=0.8):
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )


# ============================================================
# GENERIC TRAINING FUNCTION FOR SKLEARN MODELS
# ============================================================

def train_model(model, df_train, df_test, target="SO2"):
    """
    Generic training wrapper for Linear Regression, Random Forest, or XGBoost.
    Returns trained model and test data (X_test, y_test)
    """
    X_train, y_train = split_X_y(df_train, target)
    X_test, y_test = split_X_y(df_test, target)

    model.fit(X_train, y_train)

    return model, X_test, y_test


# ============================================================
# LSTM HELPERS
# ============================================================

def prepare_lstm_data(df, target="SO2", window=24):
    """
    Converts dataframe into supervised learning input (X, y) for LSTM.
    Works for univariate and multivariate.
    """
    values = df.values
    target_idx = df.columns.get_loc(target)

    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i][target_idx])

    return np.array(X), np.array(y)


def build_lstm_model(input_shape, lr=0.001, units=64):
    """
    Builds a simple LSTM Keras model.
    """
    model = Sequential([
        LSTM(units, return_sequences=False, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    return model


def train_lstm(df_train, df_test, target="SO2", window=24,
               epochs=10, batch_size=32, lr=0.001, units=64):
    """
    Full LSTM training pipeline.
    Returns trained model, training history, and test data (X_test, y_test)
    """
    # Prepare data
    X_train, y_train = prepare_lstm_data(df_train, target, window)
    X_test, y_test = prepare_lstm_data(df_test, target, window)

    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                             lr=lr, units=units)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history, X_test, y_test
