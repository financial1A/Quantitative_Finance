"""
Multi-Asset Momentum & Signal Dashboard
-----------------------------------------
Cross-asset-class (Equities, Crypto, FX, Bonds/Rates) momentum signal engine
with multi-timeframe RSI confluence, trend filter, and a weighted composite score.

Run: streamlit run multi_asset_signal_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import ta
import yfinance as yf
import plotly.graph_objs as go

st.set_page_config(page_title="Multi-Asset Momentum Dashboard", layout="wide")

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
.stApp { background-color: #f59e0b; }
section[data-testid="stSidebar"] { background-color: #131722; }
section[data-testid="stSidebar"] * { color: #d1d5db !important; }
[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; }
h1, h2, h3 { color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Multi-Asset Momentum & Signal Dashboard")
st.caption("Equities · Crypto · FX · Bonds/Rates — multi-timeframe RSI confluence with a trend filter. Research tool, not investment advice.")

# ============================================================
# ASSET UNIVERSE
# ============================================================
UNIVERSE = {
    "Crypto (Binance)": {
        "kind": "crypto",
        "tickers": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        "timeframes": ["1h", "4h", "1d"],
    },
    "Equities (Yahoo)": {
        "kind": "yf",
        "tickers": ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ"],
        "timeframes": ["1h", "1d"],
    },
    "FX (Yahoo)": {
        "kind": "yf",
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"],
        "timeframes": ["1h", "1d"],
    },
    "Bonds / Rates (Yahoo)": {
        "kind": "yf",
        "tickers": ["TLT", "IEF", "SHY", "^TNX", "^TYX"],
        "timeframes": ["1d"],
    },
}

with st.sidebar:
    st.header("Asset Selection")
    asset_class = st.selectbox("Asset Class", list(UNIVERSE.keys()))
    cfg = UNIVERSE[asset_class]
    selected_ticker = st.selectbox("Ticker", cfg["tickers"])
    timeframe = st.selectbox("Timeframe", cfg["timeframes"])
    period = st.text_input("Lookback (Yahoo only, e.g. 6mo, 1y, 2y)", value="1y")

    st.header("Signal Settings")
    trend_ma = st.slider("Trend filter — MA length (bars)", 20, 200, 50, 10)
    ob_level = st.slider("Overbought level", 60, 90, 70)
    os_level = st.slider("Oversold level", 10, 40, 30)

    st.caption(
        "Bonds/Rates note: ^TNX and ^TYX are yield indices, not price ETFs — "
        "RSI on a yield reads inversely to RSI on a bond price (TLT/IEF)."
    )

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=300)
def fetch_crypto(symbol, timeframe):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


@st.cache_data(ttl=300)
def fetch_yf(symbol, interval, period):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"datetime": "timestamp", "date": "timestamp"}, inplace=True)
    return df


if cfg["kind"] == "crypto":
    df = fetch_crypto(selected_ticker, timeframe)
else:
    df = fetch_yf(selected_ticker, timeframe, period)

if df is None or df.empty or len(df) < trend_ma + 5:
    st.error("Not enough data returned for this ticker/timeframe combination.")
    st.stop()

# ============================================================
# SIGNAL ENGINE
# ============================================================
def analyze(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["RSI_7"] = ta.momentum.RSIIndicator(close=df["close"], window=7).rsi()
    df["RSI_5"] = ta.momentum.RSIIndicator(close=df["close"], window=5).rsi()

    df["MA_trend"] = df["close"].rolling(trend_ma).mean()
    df["trend_up"] = df["close"] > df["MA_trend"]

    macd = ta.trend.MACD(close=df["close"])
    df["MACD_hist"] = macd.macd_diff()

    if "volume" in df.columns and df["volume"].notna().any():
        df["vol_avg20"] = df["volume"].rolling(20).mean()
        df["vol_confirm"] = df["volume"] > df["vol_avg20"]
    else:
        df["vol_confirm"] = np.nan

    return df


def composite_score(row) -> float:
    """
    Weighted composite in [-100, 100]:
      +/- multi-timeframe RSI confluence (50%)
      +/- trend filter alignment        (30%)
      +/- MACD histogram momentum sign  (20%)
    Positive = bullish mean-reversion/momentum setup, negative = bearish.
    """
    rsi_component = 0
    for r, w in [(row["RSI_14"], 0.5), (row["RSI_7"], 0.3), (row["RSI_5"], 0.2)]:
        if r < os_level:
            rsi_component += w * (os_level - r)
        elif r > ob_level:
            rsi_component -= w * (r - ob_level)
    rsi_component = np.clip(rsi_component * 2, -50, 50)

    trend_component = 30 if row["trend_up"] else -30

    macd_component = 20 if row["MACD_hist"] > 0 else -20 if row["MACD_hist"] < 0 else 0

    return rsi_component + trend_component * 0.6 + macd_component * 0.6  # keep total bounded ~[-100,100]


def classify(score: float, trend_up: bool) -> str:
    if score >= 45:
        return "Strong long — momentum + trend aligned" if trend_up else "Long (counter-trend, weaker conviction)"
    if score >= 20:
        return "Lean long"
    if score <= -45:
        return "Strong short — momentum + trend aligned" if not trend_up else "Short (counter-trend, weaker conviction)"
    if score <= -20:
        return "Lean short"
    return "No clear edge — inside neutral band"


df = analyze(df)
df["Composite_Score"] = df.apply(composite_score, axis=1)
df["Signal"] = df.apply(lambda r: classify(r["Composite_Score"], r["trend_up"]), axis=1)

latest = df.iloc[-1]

# ============================================================
# TOP PANEL
# ============================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Price", f"{latest['close']:.4f}" if latest["close"] < 10 else f"{latest['close']:.2f}")
with col2:
    st.metric("Composite Score", f"{latest['Composite_Score']:.0f}", delta=None)
with col3:
    st.metric("Trend Filter", f"MA{trend_ma}: {'Above ↑' if latest['trend_up'] else 'Below ↓'}")

st.subheader("Signal")
st.info(f"**{latest['Signal']}**")

with st.expander("RSI / MACD detail"):
    st.code(
        f"RSI-14: {latest['RSI_14']:.2f}\n"
        f"RSI-7:  {latest['RSI_7']:.2f}\n"
        f"RSI-5:  {latest['RSI_5']:.2f}\n"
        f"MACD hist: {latest['MACD_hist']:.4f}\n"
        f"Volume confirm: {latest['vol_confirm']}"
    )

# ============================================================
# CHARTS
# ============================================================
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", line=dict(color="#f2350a")))
fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["MA_trend"], name=f"MA{trend_ma}", line=dict(color="#f59e0b", dash="dash")))
fig_price.update_layout(template="plotly_dark", height=380, title=f"{selected_ticker} — price & trend filter")
st.plotly_chart(fig_price, use_container_width=True)

fig_rsi = go.Figure()
for col, color in zip(["RSI_14", "RSI_7", "RSI_5"], ["#ff5500", "#34d399", "#f59e0b"]):
    fig_rsi.add_trace(go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col, line=dict(color=color)))
fig_rsi.add_hline(y=ob_level, line_dash="dash", line_color="#ef4444")
fig_rsi.add_hline(y=os_level, line_dash="dash", line_color="#22c55e")
fig_rsi.update_layout(template="plotly_dark", height=350, title="Multi-timeframe RSI confluence", yaxis_title="RSI")
st.plotly_chart(fig_rsi, use_container_width=True)

fig_score = go.Figure()
fig_score.add_trace(go.Scatter(x=df["timestamp"], y=df["Composite_Score"], mode="lines", name="Composite score",
                                line=dict(color="#f3fa8b")))
fig_score.add_hline(y=45, line_dash="dot", line_color="#22c55e")
fig_score.add_hline(y=-45, line_dash="dot", line_color="#ef4444")
fig_score.update_layout(template="plotly_dark", height=300, title="Composite momentum score (bounded ±100)",
                         yaxis_title="Score")
st.plotly_chart(fig_score, use_container_width=True)

st.divider()
st.caption(
    "Composite score = weighted blend of multi-timeframe RSI mean-reversion signal (50%), "
    f"trend filter vs {trend_ma}-bar MA (30%), and MACD histogram sign (20%). "
    "Signal thresholds and weights are heuristic — validate against your own backtests before sizing trades. "
    "Not investment advice."
)
