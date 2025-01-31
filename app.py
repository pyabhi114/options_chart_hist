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
    def __init__(self, symbol, expiry_date):
        """Initialize the Breeze API connection"""
        load_dotenv()
        self.breeze = None
        self.is_connected = False
        self.symbol = symbol
        self.expiry_date = expiry_date
        self.connect()

    def connect(self):
        """Connect to Breeze API"""
        try:
            # Get credentials from environment variables
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            user_id = os.getenv("USER_ID")
            password = os.getenv("PASSWORD")
            google_auth_key = os.getenv("GOOGLE_AUTH_KEY")

            if not all([api_key, api_secret, user_id, password, google_auth_key]):
                raise ValueError("Missing required environment variables")

            # Initialize Breeze connection
            self.breeze = BreezeConnect(api_key=api_key)
            self.breeze.generate_session(api_secret=api_secret,
                                      session_token=self.breeze.get_session_token())
            self.is_connected = True
            logger.info("Successfully connected to Breeze API")
            return True

        except Exception as e:
            logger.error(f"Error connecting to Breeze API: {str(e)}")
            st.error(f"Error connecting to Breeze API: {str(e)}")
            return False

    def get_historical_data(self, strike_price, right, from_date, to_date, interval="1minute"):
        """
        Get historical option data
        
        Parameters:
        -----------
            strike_price: Strike price of the option
            right: Option type ("call" or "put")
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            interval: Data interval ("1minute" or "5minute")
        
        Returns:
        --------
            pandas.DataFrame: Historical data with OHLCV values
        """
        try:
            # Convert dates to API format
            from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
            to_datetime = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)
            expiry_datetime = self.expiry_date
            
            # Format dates for API
            from_date_str = from_datetime.strftime("%Y-%m-%dT07:00:00.000Z")
            to_date_str = to_datetime.strftime("%Y-%m-%dT07:00:00.000Z")
            expiry_date_str = expiry_datetime.strftime("%Y-%m-%d")
            
            # Get historical data
            hist_data = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=from_date_str,
                to_date=to_date_str,
                stock_code=self.symbol,
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry_date_str,
                right=right,
                strike_price=str(strike_price)
            )

            if not hist_data or 'Success' not in hist_data or not hist_data['Success']:
                raise ValueError("Failed to fetch historical data")

            # Convert to DataFrame
            df = pd.DataFrame(hist_data['Success'])
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            st.error(f"Error getting historical data: {str(e)}")
            return None

    def get_current_market_price(self):
        """Get current market price for the symbol"""
        try:
            if self.symbol == "NIFTY":
                quote = self.breeze.get_quote(stock_code="NIFTY", exchange_code="NFO", expiry=self.expiry_date.strftime("%Y-%m-%d"))
            else:  # BANKNIFTY
                quote = self.breeze.get_quote(stock_code="BANKNIFTY", exchange_code="NFO", expiry=self.expiry_date.strftime("%Y-%m-%d"))
            return float(quote['ltp'])
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return None

    def get_atm_strike(self):
        """Get ATM strike price based on current market price"""
        current_price = self.get_current_market_price()
        if current_price is None:
            return None
        
        # Round to nearest strike
        strike_diff = 50 if self.symbol == "NIFTY" else 100
        atm_strike = round(current_price / strike_diff) * strike_diff
        return atm_strike

    def get_live_atm_data(self):
        """Get live data for ATM CE and PE options"""
        atm_strike = self.get_atm_strike()
        if atm_strike is None:
            return None, None

        try:
            # Get CE data
            ce_data = self.breeze.get_quote(
                stock_code=f"{self.symbol}",
                exchange_code="NFO",
                expiry=self.expiry_date.strftime("%Y-%m-%d"),
                strike_price=atm_strike,
                right="CE"
            )
            
            # Get PE data
            pe_data = self.breeze.get_quote(
                stock_code=f"{self.symbol}",
                exchange_code="NFO",
                expiry=self.expiry_date.strftime("%Y-%m-%d"),
                strike_price=atm_strike,
                right="PE"
            )
            
            return ce_data, pe_data
        except Exception as e:
            logger.error(f"Error getting live ATM data: {e}")
            return None, None

def plot_candlestick(df, title):
    """Create a candlestick chart using plotly"""
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'])])
    
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=600
    )
    
    return fig

def main():
    st.title("Option Chain Historical Charts")
    
    # Initialize inputs
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"], index=0)
    
    expiry_date = st.date_input(
        "Expiry Date",
        datetime(2025, 2, 6).date(),
        min_value=datetime(2024, 1, 1).date(),
        max_value=datetime(2025, 12, 31).date()
    )
    
    # Initialize data fetcher
    data_fetcher = OptionDataFetcher(symbol, expiry_date)
    
    # Create sidebar inputs
    with st.sidebar:
        st.header("Chart Settings")
        
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
        
        # Interval selection
        interval = st.selectbox("Interval", ["1minute", "5minute"], index=0)
        
        # Fetch button
        fetch_button = st.button("Fetch Data")

    # Fetch and display data
    if fetch_button:
        with st.spinner('Fetching data...'):
            df = data_fetcher.get_historical_data(
                strike_price=strike_price,
                right="call" if option_type == "CE" else "put",
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                interval=interval
            )
            
            if df is not None and not df.empty:
                # Create candlestick chart
                title = f"{symbol} {strike_price} {option_type} ({interval} candles)"
                fig = plot_candlestick(df, title)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.subheader("Data Table")
                st.dataframe(df.sort_values('datetime', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
