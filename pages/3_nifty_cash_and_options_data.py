import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import os
import json
from dotenv import load_dotenv
from breeze_connect import BreezeConnect
import plotly.graph_objects as go

# --- Connect to Breeze ---
@st.cache_resource
def connect_breeze():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    try:
        with open("session_token.json", "r") as f:
            session_token = json.load(f).get("session_token")

        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        return breeze
    except Exception as e:
        st.error(f"Failed to connect to Breeze API: {e}")
        return None

# --- Common Plotting Function ---
def plot_candlestick(df, title, include_oi=False):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"))

    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        yaxis='y2', opacity=0.3))

    if include_oi:
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['open_interest'], name='Open Interest',
            yaxis='y3', line=dict(color='yellow')))

    fig.update_layout(
        template='plotly_dark',
        title=title,
        height=800,
        yaxis_title='Price',
        xaxis_title='Time',
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
    )

    if include_oi:
        fig.update_layout(
            yaxis3=dict(title="Open Interest", overlaying="y", side="right", position=0.95, showgrid=False)
        )

    return fig

# --- NIFTY Index Data Fetcher ---
def get_nifty_cash_data(breeze, from_date, to_date, interval):
    try:
        response = breeze.get_historical_data_v2(
            interval=interval,
            from_date=from_date.strftime("%Y-%m-%dT07:00:00.000Z"),
            to_date=(to_date + timedelta(days=1)).strftime("%Y-%m-%dT07:00:00.000Z"),
            stock_code="NIFTY",
            exchange_code="NSE",
            product_type="cash"
        )

        df = pd.DataFrame(response["Success"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    except Exception as e:
        st.error(f"Error fetching NIFTY data: {e}")
        return None

# --- Options Data Fetcher ---
def get_options_data(breeze, symbol, strike_price, right, expiry_date, from_date, to_date, interval):
    try:
        response = breeze.get_historical_data_v2(
            interval=interval,
            from_date=from_date.strftime("%Y-%m-%dT07:00:00.000Z"),
            to_date=(to_date + timedelta(days=1)).strftime("%Y-%m-%dT07:00:00.000Z"),
            stock_code=symbol,
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date.strftime("%Y-%m-%dT07:00:00.000Z"),
            right=right,
            strike_price=str(strike_price)
        )

        df = pd.DataFrame(response["Success"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest']:
            df[col] = df[col].astype(float)
        return df

    except Exception as e:
        st.error(f"Error fetching Options data: {e}")
        return None

# --- Streamlit UI ---
def main():
    st.set_page_config(layout="wide")
    st.title("📊 NIFTY Index & Options Chain Dashboard")

    breeze = connect_breeze()
    if not breeze:
        return

    tab1, tab2 = st.tabs(["📈 NIFTY Index Cash Chart", "📘 Options Chain Chart"])

    # --- Tab 1: NIFTY Index ---
    with tab1:
        st.subheader("NIFTY Cash Data")
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input("From", datetime.today() - timedelta(days=1), key="nifty_from")
        with col2:
            to_date = st.date_input("To", datetime.today(), key="nifty_to")

        interval = st.selectbox("Interval", ["1minute", "5minute", "1second"], key="nifty_interval")

        if st.button("Fetch NIFTY Cash Data"):
            with st.spinner("Fetching..."):
                df = get_nifty_cash_data(breeze, from_date, to_date, interval)
                if df is not None:
                    st.success("Data Loaded")
                    st.dataframe(df)
                    st.plotly_chart(plot_candlestick(df, "NIFTY Index Cash Chart"), use_container_width=True)
                    st.download_button("Download CSV", df.to_csv(index=False), "nifty_cash.csv", "text/csv")

    # --- Tab 2: Options Chain ---
    with tab2:
        st.subheader("Options Chain Candlestick Chart")
        symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
        strike_price = st.number_input("Strike Price", value=23400, step=50)
        option_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)
        expiry = st.date_input("Expiry Date", datetime.today() + timedelta(days=7))
        from_date_opt = st.date_input("From Date", datetime.today() - timedelta(days=1), key="opt_from")
        to_date_opt = st.date_input("To Date", datetime.today(), key="opt_to")
        interval_opt = st.selectbox("Interval", ["1second", "1minute", "5minute"], index=1)

        if st.button("Fetch Options Data"):
            with st.spinner("Fetching options data..."):
                df_opt = get_options_data(
                    breeze, symbol, strike_price,
                    right="call" if option_type == "CE" else "put",
                    expiry_date=expiry,
                    from_date=from_date_opt,
                    to_date=to_date_opt,
                    interval=interval_opt
                )
                if df_opt is not None:
                    title = f"{symbol} {strike_price} {option_type} ({expiry.strftime('%d-%b-%Y')})"
                    st.success("Data Loaded")
                    st.dataframe(df_opt)
                    st.plotly_chart(plot_candlestick(df_opt, title, include_oi=True), use_container_width=True)
                    st.download_button("Download CSV", df_opt.to_csv(index=False), "option_chain.csv", "text/csv")


if __name__ == "__main__":
    main()
