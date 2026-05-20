import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
 
 
def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Create features (X) and target (y) for ML model.
    We predict tomorrow's closing price.
    """
    feature_df = df.copy()
 
    # Features: technical indicators + lagged prices
    feature_df["Prev_Close"] = feature_df["Close"].shift(1)
    feature_df["Prev_Close_2"] = feature_df["Close"].shift(2)
    feature_df["Prev_Close_5"] = feature_df["Close"].shift(5)
    feature_df["Price_Change"] = feature_df["Close"].pct_change()
    feature_df["Volume_Change"] = feature_df["Volume"].pct_change()
 
    feature_df.dropna(inplace=True)
 
    feature_cols = [
        "Prev_Close", "Prev_Close_2", "Prev_Close_5",
        "Price_Change", "Volume_Change",
        "SMA_20", "SMA_50", "RSI",
        "BB_Mid", "BB_Upper", "BB_Lower",
        "MACD", "MACD_Signal"
    ]
 
    X = feature_df[feature_cols]
    y = feature_df["Close"]
 
    return X, y, feature_df
 
 
def train_model(df: pd.DataFrame) -> dict:
    """Train both Linear Regression and Random Forest, return results"""
    X, y, feature_df = prepare_features(df)
 
    # Split: 80% train, 20% test (no shuffle - time series data!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
 
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
 
    # Random Forest (usually more accurate)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # RF doesn't need scaling
    rf_preds = rf_model.predict(X_test)
 
    feature_cols = [
        "Prev_Close", "Prev_Close_2", "Prev_Close_5",
        "Price_Change", "Volume_Change",
        "SMA_20", "SMA_50", "RSI",
        "BB_Mid", "BB_Upper", "BB_Lower",
        "MACD", "MACD_Signal"
    ]
 
    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "lr_preds": lr_preds,
        "rf_preds": rf_preds,
        "lr_mae": mean_absolute_error(y_test, lr_preds),
        "rf_mae": mean_absolute_error(y_test, rf_preds),
        "lr_r2": r2_score(y_test, lr_preds),
        "rf_r2": r2_score(y_test, rf_preds),
        "feature_df": feature_df,
        "feature_cols": feature_cols
    }
 
 
def predict_next_day(results: dict) -> float:
    """Predict tomorrow's price using Random Forest"""
    feature_df = results["feature_df"]
    last_row = feature_df[results["feature_cols"]].iloc[-1].values.reshape(1, -1)
    prediction = results["rf_model"].predict(last_row)[0]
    return round(prediction, 2)
 