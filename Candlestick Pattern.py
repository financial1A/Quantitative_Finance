"""
Candlestick Pattern Dashboard
--------------------------------
Stocks & crypto candlestick pattern detection with trend context — a pattern
only gets flagged as a signal when it appears where it actually matters
(at a local high/low or against the prevailing trend), not on every bar.

Run: streamlit run candlestick_pattern_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Candlestick Pattern Dashboard", layout="wide")

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
.stApp { background-color: yellow; }
section[data-testid="stSidebar"] { background-color: #131722; }
section[data-testid="stSidebar"] * { color: #d1d5db !important; }
[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; }
h1, h2, h3 { color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Candlestick Pattern Dashboard")
st.caption("Stocks & crypto — pattern detection with trend context. Research tool, not investment advice.")

with st.expander("How to use / what's detected", expanded=False):
    st.markdown("""
**Patterns:** Hammer, Shooting Star, Bullish/Bearish Engulfing, Doji, Evening Star, Morning Star.

**Why trend context matters:** a Hammer after a downtrend is a reversal signal; the same candle
shape mid-uptrend is noise. This dashboard tags every detected pattern with the local trend
(20-bar MA slope) and only calls it a **signal** when the pattern's implied direction disagrees
with — i.e. potentially reverses — the prevailing trend.
    """)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    market_type = st.selectbox("Market", ["Crypto", "Stock"])
    if market_type == "Crypto":
        symbol = st.text_input("Crypto Pair", "BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2)
    else:
        symbol = st.text_input("Stock Symbol", "AAPL")
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "1d"], index=2)

    days_to_fetch = st.slider("Days to Analyze", 1, 30, 10)
    trend_window = st.slider("Trend MA window (bars)", 5, 50, 20)

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=300)
def fetch_crypto(symbol, timeframe, days):
    exchange = ccxt.binance()
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    cursor = since
    for _ in range(20):  # safety cap on pagination loops
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(ohlcv) < 1000:
            break
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.drop_duplicates(subset="timestamp").reset_index(drop=True)


@st.cache_data(ttl=300)
def fetch_stock(symbol, timeframe, days):
    interval_map = {"15m": "15m", "1h": "60m", "1d": "1d"}
    yf_interval = interval_map.get(timeframe, "1d")
    df = yf.download(symbol, period=f"{days}d", interval=yf_interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Flatten MultiIndex correctly for a SINGLE ticker (previous version's join()
    # produced columns like "close_aapl", breaking every downstream lookup)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.rename(columns={"datetime": "timestamp", "date": "timestamp"}, inplace=True)
    return df


if market_type == "Crypto":
    df = fetch_crypto(symbol, timeframe, days_to_fetch)
else:
    df = fetch_stock(symbol, timeframe, days_to_fetch)

if df is None or df.empty or len(df) < trend_window + 5:
    st.warning("No data found (or not enough bars) — check symbol, timeframe, or lookback.")
    st.stop()

# ============================================================
# PATTERN DETECTION
# ============================================================
def is_hammer(row):
    body = abs(row["close"] - row["open"])
    lower_shadow = (row["open"] - row["low"]) if row["open"] > row["close"] else (row["close"] - row["low"])
    upper_shadow = row["high"] - max(row["open"], row["close"])
    return lower_shadow > 2 * body and upper_shadow < body and body > 0

def is_shooting_star(row):
    body = abs(row["close"] - row["open"])
    upper_shadow = row["high"] - max(row["open"], row["close"])
    lower_shadow = min(row["open"], row["close"]) - row["low"]
    return upper_shadow > 2 * body and lower_shadow < body and body > 0

def is_doji(row):
    return abs(row["close"] - row["open"]) <= 0.1 * (row["high"] - row["low"] + 1e-9)

def is_bullish_engulfing(prev, curr):
    return (prev["close"] < prev["open"] and curr["close"] > curr["open"]
            and curr["close"] > prev["open"] and curr["open"] < prev["close"])

def is_bearish_engulfing(prev, curr):
    return (prev["close"] > prev["open"] and curr["close"] < curr["open"]
            and curr["open"] > prev["close"] and curr["close"] < prev["open"])

def is_evening_star(p1, p2, p3):
    return (p1["close"] > p1["open"]
            and abs(p2["close"] - p2["open"]) / (p2["high"] - p2["low"] + 1e-9) < 0.3
            and p3["close"] < p3["open"] and p3["close"] < p1["open"])

def is_morning_star(p1, p2, p3):
    return (p1["close"] < p1["open"]
            and abs(p2["close"] - p2["open"]) / (p2["high"] - p2["low"] + 1e-9) < 0.3
            and p3["close"] > p3["open"] and p3["close"] > p1["open"])


# Implied direction if the pattern fires as a genuine reversal
PATTERN_DIRECTION = {
    "Hammer": "bullish",
    "Shooting Star": "bearish",
    "Bullish Engulfing": "bullish",
    "Bearish Engulfing": "bearish",
    "Doji": "neutral",
    "Evening Star": "bearish",
    "Morning Star": "bullish",
}


@st.cache_data(ttl=300)
def detect_patterns(df: pd.DataFrame, trend_window: int) -> pd.DataFrame:
    df = df.copy()
    df["trend_ma"] = df["close"].rolling(trend_window).mean()
    df["trend_slope"] = df["trend_ma"].diff()
    df["trend"] = np.where(df["trend_slope"] > 0, "up", np.where(df["trend_slope"] < 0, "down", "flat"))

    records = []
    for i in range(2, len(df)):
        row, prev, pre_prev = df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
        trend = row["trend"]
        found = []
        if is_hammer(row): found.append("Hammer")
        if is_shooting_star(row): found.append("Shooting Star")
        if is_doji(row): found.append("Doji")
        if is_bullish_engulfing(prev, row): found.append("Bullish Engulfing")
        if is_bearish_engulfing(prev, row): found.append("Bearish Engulfing")
        if is_evening_star(pre_prev, prev, row): found.append("Evening Star")
        if is_morning_star(pre_prev, prev, row): found.append("Morning Star")

        for pattern in found:
            direction = PATTERN_DIRECTION[pattern]
            # A reversal signal = pattern direction opposes the prevailing trend
            is_signal = (direction == "bullish" and trend == "down") or \
                        (direction == "bearish" and trend == "up")
            records.append({
                "timestamp": row["timestamp"], "close": row["close"], "pattern": pattern,
                "direction": direction, "trend_at_time": trend, "is_reversal_signal": is_signal,
            })
    return pd.DataFrame(records)


df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
patterns_df = detect_patterns(df, trend_window)

# ============================================================
# TOP SUMMARY
# ============================================================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Bars analyzed", len(df))
with c2:
    st.metric("Patterns detected", len(patterns_df))
with c3:
    n_signals = int(patterns_df["is_reversal_signal"].sum()) if not patterns_df.empty else 0
    st.metric("Trend-context reversal signals", n_signals)

# ============================================================
# CANDLESTICK CHART WITH PATTERN MARKERS
# ============================================================
st.subheader(f"{symbol} — candlestick chart with pattern markers")
fig = go.Figure(data=[go.Candlestick(
    x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name=symbol, increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
)])
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"].rolling(trend_window).mean(),
                          name=f"MA{trend_window}", line=dict(color="#f59e0b", width=1.2)))

if not patterns_df.empty:
    signals = patterns_df[patterns_df["is_reversal_signal"]]
    bullish_sig = signals[signals["direction"] == "bullish"]
    bearish_sig = signals[signals["direction"] == "bearish"]
    fig.add_trace(go.Scatter(x=bullish_sig["timestamp"], y=bullish_sig["close"] * 0.995,
                              mode="markers", marker=dict(symbol="triangle-up", size=11, color="#22c55e"),
                              name="Bullish reversal signal", text=bullish_sig["pattern"], hoverinfo="text+x"))
    fig.add_trace(go.Scatter(x=bearish_sig["timestamp"], y=bearish_sig["close"] * 1.005,
                              mode="markers", marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                              name="Bearish reversal signal", text=bearish_sig["pattern"], hoverinfo="text+x"))

fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PATTERN FREQUENCY BY DAY
# ============================================================
if not patterns_df.empty:
    st.subheader("Pattern frequency by day")
    patterns_df["date"] = pd.to_datetime(patterns_df["timestamp"]).dt.date
    daily_counts = patterns_df.groupby(["date", "pattern"]).size().reset_index(name="count")
    pivot = daily_counts.pivot(index="date", columns="pattern", values="count").fillna(0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=[str(d) for d in pivot.index],
        colorscale="RdYlGn", text=pivot.values.astype(int), texttemplate="%{text}",
    ))
    fig_heat.update_layout(template="plotly_dark", height=350, title="All detected patterns (raw counts)")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Reversal signals only (trend-context filtered)")
    if n_signals > 0:
        st.dataframe(
            signals[["timestamp", "pattern", "direction", "trend_at_time", "close"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No trend-context reversal signals in this window — patterns fired only with the trend, or none fired.")
else:
    st.info("No candlestick patterns detected in this window.")

st.divider()
st.caption(
    "A pattern is labeled a 'reversal signal' only when its implied direction opposes the "
    f"{trend_window}-bar trend at that bar. Raw pattern counts (heatmap above) include everything, "
    "signal-filtered, for context on how selective the filter is. Not investment advice."
)
