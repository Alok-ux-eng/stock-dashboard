import yfinance as yf
import pandas as pd
 
 
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker symbol.
    ticker: e.g. 'TCS.NS' for TCS (Indian), 'AAPL' for Apple
    period: '1mo', '3mo', '6mo', '1y', '2y', '5y'
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
 
    # Keep only useful columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
 
    return df
 
 
def get_stock_info(ticker: str) -> dict:
    """Get basic company info"""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "52w_high": info.get("fiftyTwoWeekHigh", 0),
        "52w_low": info.get("fiftyTwoWeekLow", 0),
    }
 