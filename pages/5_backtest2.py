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

# File uploaders for CE and PE data, or use sample
data_file_ce = st.file_uploader("Upload Nifty ATM CE Data (CSV)", type=["csv"], key="ce")
data_file_pe = st.file_uploader("Upload Nifty ATM PE Data (CSV)", type=["csv"], key="pe")

if data_file_ce and data_file_pe:
    df_ce = pd.read_csv(data_file_ce)
    df_pe = pd.read_csv(data_file_pe)
else:
    st.info("No files uploaded. Fetching data using Breeze API (from app.py functions). Please provide the required option details below.")
    import sys
    sys.path.append('.')  # Ensure current directory is in path
    from app import connect_breeze, get_options_data

    # User input for option details
    symbol = st.text_input("Symbol", value="NIFTY")
    expiry_str = st.text_input("Expiry Date (YYYY-MM-DD)")
    strike_price = st.number_input("ATM Strike Price", value=22500)
    from_date = st.date_input("From Date", value=pd.to_datetime("2025-06-11"))
    to_date = st.date_input("To Date", value=pd.to_datetime("2025-06-11"))
    interval = st.selectbox("Interval", ["1minute", "5minute", "15minute"], index=0)

    # Only fetch if expiry is filled
    if expiry_str:
        try:
            expiry_date = pd.to_datetime(expiry_str)
            breeze = connect_breeze()
            if breeze:
                with st.spinner("Fetching CE data..."):
                    df_ce = get_options_data(
                        breeze, symbol, strike_price, "call", expiry_date, from_date, to_date, interval
                    )
                with st.spinner("Fetching PE data..."):
                    df_pe = get_options_data(
                        breeze, symbol, strike_price, "put", expiry_date, from_date, to_date, interval
                    )
                if df_ce is not None and df_pe is not None:
                    st.success("Data fetched for both CE and PE.")
                else:
                    st.error("Failed to fetch both CE and PE data. Check API credentials and parameters.")
            else:
                st.error("Could not connect to Breeze API.")
        except Exception as e:
            st.error(f"Error parsing expiry date or fetching data: {e}")
    else:
        st.warning("Please enter all option details to fetch data.")

# Ensure datetime column is datetime type
df_ce["datetime"] = pd.to_datetime(df_ce["datetime"])
df_pe["datetime"] = pd.to_datetime(df_pe["datetime"])
df_ce = df_ce.sort_values("datetime")
df_pe = df_pe.sort_values("datetime")

entry_time = time(9, 17)
exit_time_eod = time(15, 15)

# --- CE Leg ---
entry_row_ce = df_ce[df_ce["datetime"].dt.time == entry_time]
if entry_row_ce.empty:
    st.error("No 09:17 data found in the CE dataset.")
    ce_result = None
else:
    entry_idx_ce = entry_row_ce.index[0]
    entry_price_ce = entry_row_ce.iloc[0]["close"]
    stop_loss_ce = entry_price_ce * 1.8
    exit_price_ce = None
    exit_time_ce = None
    for i, row in df_ce.loc[entry_idx_ce+1:].iterrows():
        if row["high"] >= stop_loss_ce:
            exit_price_ce = stop_loss_ce
            exit_time_ce = row["datetime"]
            break
    if exit_price_ce is None:
        # Exit at 15:15
        eod_row_ce = df_ce[df_ce["datetime"].dt.time == exit_time_eod]
        if not eod_row_ce.empty:
            exit_price_ce = eod_row_ce.iloc[0]["close"]
            exit_time_ce = eod_row_ce.iloc[0]["datetime"]
        else:
            exit_price_ce = df_ce.iloc[-1]["close"]
            exit_time_ce = df_ce.iloc[-1]["datetime"]
    pnl_ce = entry_price_ce - exit_price_ce
    ce_result = dict(entry=entry_price_ce, exit=exit_price_ce, sl=stop_loss_ce, pnl=pnl_ce, exit_time=exit_time_ce, entry_time=df_ce.loc[entry_idx_ce, "datetime"])

# --- PE Leg ---
entry_row_pe = df_pe[df_pe["datetime"].dt.time == entry_time]
if entry_row_pe.empty:
    st.error("No 09:17 data found in the PE dataset.")
    pe_result = None
else:
    entry_idx_pe = entry_row_pe.index[0]
    entry_price_pe = entry_row_pe.iloc[0]["close"]
    stop_loss_pe = entry_price_pe * 1.8
    exit_price_pe = None
    exit_time_pe = None
    for i, row in df_pe.loc[entry_idx_pe+1:].iterrows():
        if row["high"] >= stop_loss_pe:
            exit_price_pe = stop_loss_pe
            exit_time_pe = row["datetime"]
            break
    if exit_price_pe is None:
        # Exit at 15:15
        eod_row_pe = df_pe[df_pe["datetime"].dt.time == exit_time_eod]
        if not eod_row_pe.empty:
            exit_price_pe = eod_row_pe.iloc[0]["close"]
            exit_time_pe = eod_row_pe.iloc[0]["datetime"]
        else:
            exit_price_pe = df_pe.iloc[-1]["close"]
            exit_time_pe = df_pe.iloc[-1]["datetime"]
    pnl_pe = entry_price_pe - exit_price_pe
    pe_result = dict(entry=entry_price_pe, exit=exit_price_pe, sl=stop_loss_pe, pnl=pnl_pe, exit_time=exit_time_pe, entry_time=df_pe.loc[entry_idx_pe, "datetime"])

# --- Display Results ---
if ce_result and pe_result:
    st.subheader("Backtest Results for 1 Day")
    st.write(f"**CE Entry Price (09:17):** {ce_result['entry']:.2f}")
    st.write(f"**CE Exit Price:** {ce_result['exit']:.2f} ({'Stop Loss' if ce_result['exit']==ce_result['sl'] else 'EOD'})")
    st.write(f"**CE P&L:** {ce_result['pnl']:.2f}")
    st.write(f"**CE Exit Time:** {ce_result['exit_time']}")
    st.write("---")
    st.write(f"**PE Entry Price (09:17):** {pe_result['entry']:.2f}")
    st.write(f"**PE Exit Price:** {pe_result['exit']:.2f} ({'Stop Loss' if pe_result['exit']==pe_result['sl'] else 'EOD'})")
    st.write(f"**PE P&L:** {pe_result['pnl']:.2f}")
    st.write(f"**PE Exit Time:** {pe_result['exit_time']}")
    st.write("---")
    st.write(f"**Total P&L:** {ce_result['pnl'] + pe_result['pnl']:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_ce["datetime"], df_ce["close"], label="CE Close Price", color="blue")
    ax.plot(df_pe["datetime"], df_pe["close"], label="PE Close Price", color="purple")
    ax.axvline(ce_result['entry_time'], color="green", linestyle="--", label="Entry (09:17)")
    ax.axhline(ce_result['sl'], color="red", linestyle=":", label="CE Stop Loss")
    ax.axhline(pe_result['sl'], color="orange", linestyle=":", label="PE Stop Loss")
    ax.scatter([ce_result['exit_time']], [ce_result['exit']], color="cyan", label="CE Exit", zorder=5)
    ax.scatter([pe_result['exit_time']], [pe_result['exit']], color="magenta", label="PE Exit", zorder=5)
    ax.set_title("ATM CE & PE Price with Entry, SL, and Exit")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
