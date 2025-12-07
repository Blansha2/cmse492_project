"""
cross_validation.py

Time-series cross-validation + hyperparameter tuning
for univariate & multivariate forecasting.

Supports:
- Linear Regression
- Random Forest
- XGBoost
- LSTM (Keras)
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam



# ============================================================
#  DATA SPLITTING HELPERS
# ============================================================

def split_X_y(df, target="SO2"):
    """Splits dataframe into X and y."""
    X = df.drop(columns=[target])
    
    # Remove any datetime columns — regardless of frequency
    datetime_cols = X.select_dtypes(include=["datetime", "datetime64"]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=datetime_cols)
    
    y = df[target]
    return X, y


def time_series_split(df, n_splits=5):
    """
    Expanding-window walk-forward splits.

    Example:
        TRAIN 0:   0 → 50    | TEST 50 → 60
        TRAIN 1:   0 → 60    | TEST 60 → 70
        TRAIN 2:   0 → 70    | TEST 70 → 80
    """
    total_len = len(df)
    test_size = total_len // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * test_size
        test_end = train_end + test_size

        if test_end > total_len:
            break

        splits.append((slice(0, train_end), slice(train_end, test_end)))

    return splits



# ============================================================
#  MODEL BUILDERS
# ============================================================

def build_linear_regression(**params):
    return LinearRegression(**params)


def build_random_forest(**params):
    return RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **params
    )


def build_xgb(**params):
    return XGBRegressor(
        random_state=42,
        n_jobs=-1,
        **params
    )



# ============================================================
#  GENERIC CROSS VALIDATION FOR SKLEARN MODELS
# ============================================================

def ts_cross_val_sklearn(model_builder, param_grid, df, target="SO2", n_splits=5):
    """
    Time-series CV + hyperparameter tuning for sklearn regression models.

    model_builder: e.g. build_random_forest
    param_grid: e.g. {"n_estimators": [100, 300], "max_depth": [5, 10]}
    """

    # Expand grid manually
    from itertools import product
    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())
    param_combos = [dict(zip(keys, v)) for v in product(*value_lists)]

    splits = time_series_split(df, n_splits=n_splits)

    results = []

    for params in param_combos:
        fold_scores = []

        for train_idx, test_idx in splits:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            X_train, y_train = split_X_y(df_train, target)
            X_test, y_test = split_X_y(df_test, target)

            model = model_builder(**params)
            #print("X_train dtypes:\n", X_train.dtypes)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            fold_scores.append(mse)

        results.append({
            "params": params,
            "scores": fold_scores,
            "mean_mse": np.mean(fold_scores)
        })

    # return best
    best_result = min(results, key=lambda x: x["mean_mse"])
    return best_result, results



# ============================================================
#  LSTM HELPERS + LSTM CROSS VALIDATION
# ============================================================

def prepare_lstm_data(df, target="SO2", window=24):
    values = df.values
    target_idx = df.columns.get_loc(target)

    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i][target_idx])

    return np.array(X), np.array(y)


def build_lstm_model(input_shape, lr=0.001, units=64):
    model = Sequential([
        LSTM(units, return_sequences=False, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def ts_cross_val_lstm(df, target="SO2", n_splits=3,
                      window=24,
                      param_grid={"lr": [0.001], "units": [32, 64]},
                      epochs=5,
                      batch_size=32):
    """
    Time-series cross-validation for LSTM.
    More expensive than sklearn models.
    """

    # make param combinations
    from itertools import product
    keys = list(param_grid.keys())
    vals = list(param_grid.values())
    combos = [dict(zip(keys, v)) for v in product(*vals)]

    splits = time_series_split(df, n_splits=n_splits)

    results = []

    for params in combos:
        fold_scores = []

        for train_idx, test_idx in splits:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            X_train, y_train = prepare_lstm_data(df_train, target, window)
            X_test, y_test = prepare_lstm_data(df_test, target, window)

            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lr=params["lr"],
                units=params["units"]
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            preds = model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, preds)
            fold_scores.append(mse)

        results.append({
            "params": params,
            "scores": fold_scores,
            "mean_mse": np.mean(fold_scores)
        })

    best_result = min(results, key=lambda x: x["mean_mse"])
    return best_result, results