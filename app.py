import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionDataFetcher:
    def __init__(self):
        """Initialize the Breeze API connection"""
        load_dotenv()
        self.breeze = None
        self.is_connected = False
        self.connect()

    def connect(self):
        """Connect to Breeze API"""
        try:
            # Load environment variables
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            
            if not api_key or not api_secret:
                st.error("API credentials not found in environment variables")
                return False
            
            self.breeze = BreezeConnect(api_key=api_key)
            
            # Load session token from file
            try:
                with open("session_token.json", "r") as f:
                    import json
                    session_data = json.load(f)
                    session_token = session_data.get("session_token")
                    if not session_token:
                        st.error("Session token not found in file")
                        return False
            except Exception as e:
                st.error(f"Error loading session token: {str(e)}")
                return False
            
            # Connect using session token
            self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
            self.is_connected = True
            logger.info("Successfully connected to Breeze API")
            return True
            
        except Exception as e:
            st.error(f"Error connecting to Breeze API: {str(e)}")
            return False

    def get_historical_data(self, symbol, strike_price, right, expiry_date, from_date, to_date, interval="1minute"):
        """
        Get historical option data
        
        Args:
            symbol: Stock/Index symbol (e.g., "NIFTY")
            strike_price: Strike price of the option
            right: Option type ("call" or "put")
            expiry_date: Expiry date in YYYY-MM-DD format
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            interval: Data interval ("1minute" or "5minute")
            
        Returns:
            DataFrame with historical data or None if error
        """
        try:
            if not self.is_connected:
                if not self.connect():
                    return None

            # Convert dates to API format
            from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
            to_datetime = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)
            expiry_datetime = datetime.strptime(expiry_date, "%Y-%m-%d")
            
            # Format dates for API
            from_date_str = from_datetime.strftime("%Y-%m-%dT07:00:00.000Z")
            to_date_str = to_datetime.strftime("%Y-%m-%dT07:00:00.000Z")
            expiry_date_str = expiry_datetime.strftime("%Y-%m-%dT07:00:00.000Z")
            
            # Get historical data
            hist_data = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=from_date_str,
                to_date=to_date_str,
                stock_code=symbol,
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry_date_str,
                right=right.lower(),
                strike_price=str(strike_price)
            )
            
            if not isinstance(hist_data, dict) or 'Success' not in hist_data or not hist_data['Success']:
                st.error(f"Failed to get historical data: {hist_data}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(hist_data['Success'])
            if df.empty:
                st.error("No historical data returned")
                return None
                
            # Convert data types
            df['datetime'] = pd.to_datetime(df['datetime'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            st.error(f"Error getting historical data: {str(e)}")
            return None

def plot_candlestick(df, title):
    """Create a candlestick chart using plotly"""
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'])])
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Time',
        template='plotly_dark',
        height=800
    )
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(x=df['datetime'], y=df['volume'], name='Volume', yaxis='y2', opacity=0.3)
    )
    
    # Add OI as line on third y-axis
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['open_interest'], name='Open Interest', yaxis='y3', line=dict(color='yellow'))
    )
    
    # Update layout for multiple y-axes
    fig.update_layout(
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        yaxis3=dict(
            title="Open Interest",
            overlaying="y",
            side="right",
            position=0.95,
            showgrid=False
        )
    )
    
    return fig

def main():
    st.set_page_config(page_title="Option Chain Charts", layout="wide")
    st.title("Option Chain Candlestick Charts")
    st.write("By Abhishek Gogna - Professional Algo-Based Quantitative Research Analyst")
    st.write("abhishekgogna36@gmail.com")
    
    
    # Initialize data fetcher
    data_fetcher = OptionDataFetcher()
    
    # Create sidebar inputs
    with st.sidebar:
        st.header("Chart Settings")
        
        # Symbol selection
        symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"], index=0)
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input(
                "From Date",
                datetime(2025, 1, 31).date(),
                min_value=datetime(2024, 1, 1).date(),
                max_value=datetime(2025, 12, 31).date()
            )
        with col2:
            to_date = st.date_input(
                "To Date",
                datetime(2025, 1, 31).date(),
                min_value=datetime(2024, 1, 1).date(),
                max_value=datetime(2025, 12, 31).date()
            )
        
        
        # Strike price input
        strike_price = st.number_input("Strike Price", value=23400, step=100)
        
        # Option type selection
        option_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)
        
        # Expiry date selection (you might want to make this dynamic based on available expiries)
        # Expiry date selection
        expiry_date = st.date_input(
            "Expiry Date",
            datetime(2025, 2, 6).date(),
            min_value=datetime(2024, 1, 1).date(),
            max_value=datetime(2025, 12, 31).date()
        )
        
        # Interval selection
        interval = st.selectbox("Interval", ["1minute", "5minute"], index=0)
        
        # Add a fetch button
        fetch_button = st.button("Fetch Data")

    # Main content
    if fetch_button:
        with st.spinner('Fetching data...'):
            df = data_fetcher.get_historical_data(
                symbol=symbol,
                strike_price=strike_price,
                right="call" if option_type == "CE" else "put",
                expiry_date=expiry_date.strftime("%Y-%m-%d"),
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                interval=interval
            )
            
            if df is not None:
                # Create title
                title = f"{symbol} {strike_price} {option_type} ({expiry_date.strftime('%d-%b-%Y')}) - {interval} Chart"
                
                # Create and display chart
                fig = plot_candlestick(df, title)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.subheader("Data Table")
                st.dataframe(df.sort_values('datetime', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
