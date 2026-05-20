import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import BytesIO
 
from data_fetcher import get_stock_data, get_stock_info
from indicators import add_all_indicators
from predictor import train_model, predict_next_day
 
# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)
 
st.title("📈 Stock Market Dashboard & Price Predictor")
st.markdown("Real-time stock analysis with ML-powered price prediction")

# Small UI tweaks: center title and tighten spacing
st.markdown(
    "<style>\n"
    "header .css-1v3fvcr {padding: 0.5rem 1rem;}\n"
    "h1 {text-align: left;}\n"
    ".stMetric {margin-bottom: 0.25rem;}\n"
    "</style>", unsafe_allow_html=True
)
 
# ── Sidebar Controls ──────────────────────────────────────────
st.sidebar.header("Settings")
 
# Popular Indian + US stocks
popular_stocks = {
    "TCS (Tata Consultancy)": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Wipro": "WIPRO.NS",
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
}
 
selected_name = st.sidebar.selectbox("Choose a stock", list(popular_stocks.keys()))
ticker = popular_stocks[selected_name]
 
custom_ticker = st.sidebar.text_input("Or enter custom ticker (e.g. SBIN.NS)")
if custom_ticker:
    ticker = custom_ticker.upper()
 
period = st.sidebar.selectbox(
    "Time Period",
    ["3mo", "6mo", "1y", "2y"],
    index=2
)
 
# ── Load Data ─────────────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(ticker, period):
    df = get_stock_data(ticker, period)
    df = add_all_indicators(df)
    return df
 
with st.spinner("Fetching live data..."):
    try:
        df = load_data(ticker, period)
        info = get_stock_info(ticker)
    except Exception as e:
        st.error(f"Could not fetch data for {ticker}. Try another ticker.")
        st.stop()
 
# ── Metric Cards ──────────────────────────────────────────────
current_price = df["Close"].iloc[-1]
prev_price = df["Close"].iloc[-2]
price_change = current_price - prev_price
pct_change = (price_change / prev_price) * 100

# Wider first column for the main metric
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

# Show optional logo if available in `info`
with col1:
    try:
        logo_url = info.get("logo_url") or info.get("logo")
    except Exception:
        logo_url = None

    if logo_url:
        st.image(logo_url, width=72)

    # Main price metric
    delta_label = f"{pct_change:.2f}%"
    col1.metric("Current Price", f"${current_price:.2f}", delta_label)

col2.metric("52W High", f"${info.get('52w_high', 0):.2f}")
col3.metric("52W Low", f"${info.get('52w_low', 0):.2f}")
col4.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
col5.metric("Sector", info.get("sector", "N/A"))

# Small utilities: download CSV and last update
st.markdown("---")
csv = df.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")
st.caption(f"Last updated: {df.index[-1]}")
 
st.markdown("---")
 
# ── Candlestick Chart ─────────────────────────────────────────
st.subheader("Price History with Moving Averages")
 
fig = go.Figure()
 
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="Price"
))
 
fig.add_trace(go.Scatter(
    x=df.index, y=df["SMA_20"],
    line=dict(color="orange", width=1.5),
    name="SMA 20"
))
 
fig.add_trace(go.Scatter(
    x=df.index, y=df["SMA_50"],
    line=dict(color="blue", width=1.5),
    name="SMA 50"
))
 
# Bollinger Bands
fig.add_trace(go.Scatter(
    x=df.index, y=df["BB_Upper"],
    line=dict(color="gray", width=1, dash="dot"),
    name="BB Upper", opacity=0.5
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["BB_Lower"],
    line=dict(color="gray", width=1, dash="dot"),
    name="BB Lower", opacity=0.5,
    fill="tonexty", fillcolor="rgba(128,128,128,0.1)"
))
 
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)
 
# ── Volume Chart ──────────────────────────────────────────────
st.subheader("Trading Volume")
vol_fig = px.bar(df, x=df.index, y="Volume", color_discrete_sequence=["#636EFA"])
vol_fig.update_layout(height=250, template="plotly_white")
st.plotly_chart(vol_fig, use_container_width=True)
 
# ── RSI + MACD ────────────────────────────────────────────────
col_rsi, col_macd = st.columns(2)
 
with col_rsi:
    st.subheader("RSI Indicator")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    rsi_fig.update_layout(height=300, template="plotly_white", yaxis=dict(range=[0, 100]))
    st.plotly_chart(rsi_fig, use_container_width=True)
 
with col_macd:
    st.subheader("MACD Indicator")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="orange")))
    macd_fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color="gray", opacity=0.5))
    macd_fig.update_layout(height=300, template="plotly_white")
    st.plotly_chart(macd_fig, use_container_width=True)
 
# ── ML Price Prediction ───────────────────────────────────────
st.markdown("---")
st.subheader("🤖 ML Price Prediction")
 
if st.button("Run Prediction Model"):
    with st.spinner("Training model on historical data..."):
        results = train_model(df)
        next_price = predict_next_day(results)
 
    direction = "📈 UP" if next_price > current_price else "📉 DOWN"
    change = next_price - current_price
 
    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
    col_pred1.metric("Predicted Next Price", f"${next_price:.2f}", f"{change:.2f}")
    col_pred2.metric("Direction", direction)
    col_pred3.metric("Random Forest R²", f"{results['rf_r2']:.3f}")
    col_pred4.metric("RF Mean Abs Error", f"${results['rf_mae']:.2f}")
 
    # Actual vs Predicted chart
    st.subheader("Actual vs Predicted Prices (Test Set)")
    pred_df = pd.DataFrame({
        "Actual": results["y_test"].values,
        "Linear Regression": results["lr_preds"],
        "Random Forest": results["rf_preds"]
    }, index=results["y_test"].index)
 
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Actual"], name="Actual", line=dict(color="black", width=2)))
    pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Linear Regression"], name="Linear Regression", line=dict(color="blue", dash="dash")))
    pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Random Forest"], name="Random Forest", line=dict(color="green", dash="dash")))
    pred_fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(pred_fig, use_container_width=True)
 
    # Model comparison table
    st.subheader("Model Performance Comparison")
    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score": [f"{results['lr_r2']:.4f}", f"{results['rf_r2']:.4f}"],
        "Mean Absolute Error": [f"${results['lr_mae']:.2f}", f"${results['rf_mae']:.2f}"],
    })
    st.dataframe(comparison, use_container_width=True)
 
# ── Raw Data ──────────────────────────────────────────────────
with st.expander("View Raw Data"):
    st.dataframe(df.tail(30), use_container_width=True)
 