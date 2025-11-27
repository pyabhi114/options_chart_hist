
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import os
import json
from dotenv import load_dotenv
from breeze_connect import BreezeConnect
import plotly.graph_objects as go

def connect_breeze():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        st.error("API key or secret not found in .env file")
        return None

    try:
        with open("session_token.json", "r") as f:
            session_token = json.load(f).get("session_token")

        if not session_token:
            st.error("Session token not found in session_token.json")
            return None

        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        return breeze
    except Exception as e:
        st.error(f"Error connecting to Breeze API: {e}")
        return None

def get_nifty_cash_data(breeze, from_date, to_date, interval):
    from_str = from_date.strftime("%Y-%m-%dT07:00:00.000Z")
    to_str = (to_date + timedelta(days=1)).strftime("%Y-%m-%dT07:00:00.000Z")

    try:
        data = breeze.get_historical_data_v2(
            interval=interval,
            from_date=from_str,
            to_date=to_str,
            stock_code="INDVIX",
            exchange_code="NSE",
            product_type="cash"
        )

        if "Success" not in data:
            st.error(f"Error fetching data: {data}")
            return None

        df = pd.DataFrame(data["Success"])
        if df.empty:
            st.warning("No data returned.")
            return None

        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df
    except Exception as e:
        st.error(f"Exception occurred: {e}")
        return None

def plot_candlestick(df, title):
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Time',
        template='plotly_dark',
        height=800
    )

    fig.add_trace(go.Bar(
        x=df['datetime'],
        y=df['volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    fig.update_layout(
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

    return fig

def main():
    st.title("📈 NIFTY Index Data Downloader")

    from_date = st.date_input("From Date", datetime.today() - timedelta(days=1))
    to_date = st.date_input("To Date", datetime.today())
    interval = st.selectbox("Interval", ["1minute", "5minute", "1second"], index=0)

    if st.button("Fetch INDIA VIX Data"):
        with st.spinner("Connecting to Breeze API..."):
            breeze = connect_breeze()
            if breeze:
                df = get_nifty_cash_data(breeze, from_date, to_date, interval)
                if df is not None:
                    st.success("Data fetched successfully!")
                    st.dataframe(df)

                    # Plot chart
                    fig = plot_candlestick(df, f"NIFTY Index - {interval} Chart")
                    st.plotly_chart(fig, use_container_width=True)

                    # Download
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, "nifty_cash_data.csv", "text/csv")


if __name__ == "__main__":
    main()
