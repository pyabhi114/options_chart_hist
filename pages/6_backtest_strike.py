import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta

st.title("📊 Nifty ATM Option Sell Backtest")

st.markdown("""
Intraday Options Strategy Backtest:
- Sell ATM CE, PE, or Both at a configurable entry time
- Stop Loss % is adjustable
- Backtest across multiple days
""")

# --- User Inputs ---
strategy_choice = st.sidebar.selectbox("Option Leg", ["Both CE & PE", "Only CE", "Only PE"])
entry_time = st.sidebar.time_input("Entry Time", value=time(9, 17))
exit_time_eod = time(15, 15)
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=10, max_value=100, value=80) / 100

# --- File Upload ---
data_file_ce = st.file_uploader("Upload ATM CE Data", type="csv")
data_file_pe = st.file_uploader("Upload ATM PE Data", type="csv")

if (strategy_choice != "Only PE" and not data_file_ce) or (strategy_choice != "Only CE" and not data_file_pe):
    st.warning("Upload both CE and PE data if using 'Both' strategy.")
    st.stop()

# --- Load Data ---
df_ce = pd.read_csv(data_file_ce) if data_file_ce else None
df_pe = pd.read_csv(data_file_pe) if data_file_pe else None

def preprocess(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df['date'] = df['datetime'].dt.date
    return df

if df_ce is not None: df_ce = preprocess(df_ce)
if df_pe is not None: df_pe = preprocess(df_pe)

# --- Backtest Logic ---
def run_leg_backtest(df, entry_time, sl_pct):
    daily_results = []
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        entry_row = day_data[day_data['datetime'].dt.time == entry_time]
        if entry_row.empty:
            continue
        entry_price = entry_row.iloc[0]['close']
        sl_price = entry_price * (1 + sl_pct)
        exit_price, exit_time = None, None
        entry_idx = entry_row.index[0]
        for i, row in day_data.loc[entry_idx+1:].iterrows():
            if row['high'] >= sl_price:
                exit_price = sl_price
                exit_time = row['datetime']
                break
        if exit_price is None:
            eod_row = day_data[day_data['datetime'].dt.time == exit_time_eod]
            exit_price = eod_row.iloc[0]['close'] if not eod_row.empty else day_data.iloc[-1]['close']
            exit_time = eod_row.iloc[0]['datetime'] if not eod_row.empty else day_data.iloc[-1]['datetime']
        pnl = entry_price - exit_price
        daily_results.append({
            "date": date,
            "entry": entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "exit_time": exit_time
        })
    return pd.DataFrame(daily_results)

results_ce = run_leg_backtest(df_ce, entry_time, stop_loss_pct) if strategy_choice != "Only PE" else pd.DataFrame()
results_pe = run_leg_backtest(df_pe, entry_time, stop_loss_pct) if strategy_choice != "Only CE" else pd.DataFrame()

# --- Combine Results ---
combined = pd.DataFrame()
if not results_ce.empty: combined["CE P&L"] = results_ce["pnl"]
if not results_pe.empty: combined["PE P&L"] = results_pe["pnl"]
if not combined.empty: 
    combined["Total P&L"] = combined.sum(axis=1)

# --- Display Results ---
if not combined.empty:
    st.subheader("📈 Backtest Summary")
    st.write(f"**Total Days Backtested:** {len(combined)}")
    st.write(f"**Total Strategy P&L:** ₹{combined['Total P&L'].sum():.2f}")
    st.write(f"**Average Daily P&L:** ₹{combined['Total P&L'].mean():.2f}")
    st.write(f"**Winning Days:** {(combined['Total P&L'] > 0).sum()} / {len(combined)}")
    st.write(f"**Losing Days:** {(combined['Total P&L'] < 0).sum()} / {len(combined)}")

    # P&L Over Time Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    combined["Total P&L"].cumsum().plot(ax=ax, label="Cumulative P&L", color="green")
    ax.set_title("Cumulative P&L Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative P&L")
    ax.legend()
    st.pyplot(fig)

    # Table
    st.subheader("📅 Daily Backtest Results")
    st.dataframe(combined.reset_index(drop=True))

else:
    st.info("No backtest results to show. Ensure data has 09:17 timestamp and valid structure.")
