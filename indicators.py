import pandas as pd
import numpy as np
 
 
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Simple Moving Average for 20 and 50 days"""
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    return df
 
 
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI - Relative Strength Index (0-100)
    Above 70 = overbought, Below 30 = oversold
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
 
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
 
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df
 
 
def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands - shows price volatility
    Price touching upper band = possibly overbought
    Price touching lower band = possibly oversold
    """
    df["BB_Mid"] = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    df["BB_Upper"] = df["BB_Mid"] + (2 * std)
    df["BB_Lower"] = df["BB_Mid"] - (2 * std)
    return df
 
 
def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD - Moving Average Convergence Divergence
    Trend-following momentum indicator
    """
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df
 
 
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators at once"""
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df.dropna(inplace=True)
    return df