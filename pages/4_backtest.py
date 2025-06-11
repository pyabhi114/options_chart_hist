import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

st.title("Nifty ATM Option Sell Backtest (09:17 Sell, 80% SL)")

st.markdown("""
This app simulates a simple intraday options strategy:
- At 09:17, sell the ATM option of Nifty.
- Set a stop loss at 80% above the sell price.
- Track P&L for the day and plot the results.
""")

# File uploader for user data, or use sample
data_file = st.file_uploader("Upload Nifty Options Data (CSV)", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)
else:
    st.info("No file uploaded. Using sample data.")
    # Sample data: Simulated 1-min OHLC for ATM option
    times = pd.date_range("2025-06-11 09:15", "2025-06-11 15:30", freq="1min")
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(len(times))) + 100
    df = pd.DataFrame({
        "datetime": times,
        "open": prices + np.random.uniform(-0.5, 0.5, len(times)),
        "high": prices + np.random.uniform(0, 1, len(times)),
        "low": prices - np.random.uniform(0, 1, len(times)),
        "close": prices + np.random.uniform(-0.5, 0.5, len(times)),
    })

# Ensure datetime column is datetime type
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# Find 09:17 bar
entry_time = time(9, 17)
entry_row = df[df["datetime"].dt.time == entry_time]
if entry_row.empty:
    st.error("No 09:17 data found in the dataset.")
else:
    entry_idx = entry_row.index[0]
    entry_price = entry_row.iloc[0]["close"]
    stop_loss = entry_price * 1.8

    # Simulate trade: Short at 09:17, exit at SL or EOD
    exit_price = None
    exit_time = None
    for i, row in df.loc[entry_idx+1:].iterrows():
        if row["high"] >= stop_loss:
            exit_price = stop_loss
            exit_time = row["datetime"]
            break
    if exit_price is None:
        # Exit at close
        exit_price = df.iloc[-1]["close"]
        exit_time = df.iloc[-1]["datetime"]

    pnl = entry_price - exit_price
    st.write(f"**Entry Price (09:17):** {entry_price:.2f}")
    st.write(f"**Exit Price:** {exit_price:.2f} ({'Stop Loss' if exit_price==stop_loss else 'EOD'})")
    st.write(f"**P&L:** {pnl:.2f}")
    st.write(f"**Exit Time:** {exit_time}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["datetime"], df["close"], label="Close Price")
    ax.axvline(df.loc[entry_idx, "datetime"], color="green", linestyle="--", label="Entry (09:17)")
    ax.axhline(stop_loss, color="red", linestyle=":", label="Stop Loss")
    ax.scatter([exit_time], [exit_price], color="orange", label="Exit")
    ax.set_title("ATM Option Price with Entry, SL, and Exit")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
