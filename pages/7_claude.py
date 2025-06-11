import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import os
import json
from typing import Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Nifty ATM Option Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OptionsBacktester:
    def __init__(self):
        self.entry_time = time(9, 17)
        self.exit_time_eod = time(15, 15)
        self.stop_loss_multiplier = 1.8
    
    def validate_data(self, df: pd.DataFrame, data_type: str) -> bool:
        """Validate the data structure and content"""
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        if df is None or df.empty:
            st.error(f"{data_type} data is empty")
            return False
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"{data_type} data missing columns: {missing_cols}")
            return False
        
        return True
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0.0
        
        # Remove rows with invalid data
        df = df.dropna(subset=['datetime', 'close'])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        return df
    
    def execute_trade(self, df: pd.DataFrame, option_type: str) -> Optional[Dict]:
        """Execute a single option trade with stop loss logic"""
        # Find entry point
        entry_mask = df["datetime"].dt.time == self.entry_time
        entry_rows = df[entry_mask]
        
        if entry_rows.empty:
            st.warning(f"No {self.entry_time} data found for {option_type}")
            return None
        
        entry_idx = entry_rows.index[0]
        entry_price = entry_rows.iloc[0]["close"]
        stop_loss_price = entry_price * self.stop_loss_multiplier
        
        # Track for stop loss hit
        exit_price = None
        exit_time = None
        exit_reason = "EOD"
        
        # Check for stop loss hit after entry
        for idx in range(entry_idx + 1, len(df)):
            row = df.iloc[idx]
            if row["high"] >= stop_loss_price:
                exit_price = stop_loss_price
                exit_time = row["datetime"]
                exit_reason = "Stop Loss"
                break
        
        # If no stop loss hit, exit at EOD
        if exit_price is None:
            eod_mask = df["datetime"].dt.time == self.exit_time_eod
            eod_rows = df[eod_mask]
            
            if not eod_rows.empty:
                exit_price = eod_rows.iloc[0]["close"]
                exit_time = eod_rows.iloc[0]["datetime"]
            else:
                # Use last available price
                exit_price = df.iloc[-1]["close"]
                exit_time = df.iloc[-1]["datetime"]
        
        pnl = entry_price - exit_price  # Profit from selling
        
        return {
            'option_type': option_type,
            'entry_time': entry_rows.iloc[0]["datetime"],
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'stop_loss_price': stop_loss_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'return_pct': (pnl / entry_price) * 100
        }
    
    def run_backtest(self, df_ce: pd.DataFrame, df_pe: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Run backtest for both CE and PE options"""
        ce_result = self.execute_trade(df_ce, "CE")
        pe_result = self.execute_trade(df_pe, "PE")
        
        return ce_result, pe_result

def load_api_connection():
    """Load Breeze API connection"""
    try:
        from breeze_connect import BreezeConnect
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        
        if not api_key or not api_secret:
            st.error("API credentials not found in environment variables")
            return None
        
        session_token = None
        if os.path.exists("session_token.json"):
            with open("session_token.json", "r") as f:
                session_token = json.load(f).get("session_token")
        
        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        return breeze
    
    except ImportError:
        st.error("Breeze Connect library not installed. Please install it using: pip install breeze-connect")
        return None
    except Exception as e:
        st.error(f"Error connecting to Breeze API: {e}")
        return None

def fetch_option_data(breeze, symbol: str, strike_price: int, right: str, 
                     expiry_date: datetime, from_date: datetime, to_date: datetime, 
                     interval: str) -> Optional[pd.DataFrame]:
    """Fetch option data from Breeze API"""
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
        
        if "Success" not in response or not response["Success"]:
            st.error(f"No data returned for {symbol} {strike_price} {right}")
            return None
        
        df = pd.DataFrame(response["Success"])
        return df
    
    except Exception as e:
        st.error(f"Error fetching {right} data: {e}")
        return None

def create_interactive_plot(df_ce: pd.DataFrame, df_pe: pd.DataFrame, 
                          ce_result: Dict, pe_result: Dict):
    """Create interactive plotly chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('CE Option Price', 'PE Option Price'),
        vertical_spacing=0.1
    )
    
    # CE Plot
    fig.add_trace(
        go.Scatter(x=df_ce['datetime'], y=df_ce['close'], 
                  name='CE Close', line=dict(color='blue')),
        row=1, col=1
    )
    
    # PE Plot
    fig.add_trace(
        go.Scatter(x=df_pe['datetime'], y=df_pe['close'], 
                  name='PE Close', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Add entry and exit markers
    for i, (result, row_num) in enumerate([(ce_result, 1), (pe_result, 2)]):
        if result:
            # Entry marker
            fig.add_trace(
                go.Scatter(x=[result['entry_time']], y=[result['entry_price']],
                          mode='markers', name=f"{result['option_type']} Entry",
                          marker=dict(color='green', size=10, symbol='circle')),
                row=row_num, col=1
            )
            
            # Exit marker
            fig.add_trace(
                go.Scatter(x=[result['exit_time']], y=[result['exit_price']],
                          mode='markers', name=f"{result['option_type']} Exit",
                          marker=dict(color='red', size=10, symbol='x')),
                row=row_num, col=1
            )
            
            # Stop loss line
            fig.add_hline(y=result['stop_loss_price'], 
                         line_dash="dot", line_color="red",
                         annotation_text=f"{result['option_type']} SL",
                         row=row_num, col=1)
    
    fig.update_layout(height=800, title_text="Options Backtest Results")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price")
    
    return fig

def display_results(ce_result: Dict, pe_result: Dict):
    """Display backtest results in a formatted way"""
    st.subheader("📊 Backtest Results")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    if ce_result:
        with col1:
            st.metric(
                label="CE P&L",
                value=f"₹{ce_result['pnl']:.2f}",
                delta=f"{ce_result['return_pct']:.2f}%"
            )
            
        with col2:
            st.metric(
                label="CE Entry → Exit",
                value=f"₹{ce_result['entry_price']:.2f} → ₹{ce_result['exit_price']:.2f}",
                delta=ce_result['exit_reason']
            )
    
    if pe_result:
        with col1 if not ce_result else col3:
            st.metric(
                label="PE P&L",
                value=f"₹{pe_result['pnl']:.2f}",
                delta=f"{pe_result['return_pct']:.2f}%"
            )
            
        with col2 if not ce_result else col1:
            st.metric(
                label="PE Entry → Exit",
                value=f"₹{pe_result['entry_price']:.2f} → ₹{pe_result['exit_price']:.2f}",
                delta=pe_result['exit_reason']
            )
    
    # Combined metrics
    if ce_result and pe_result:
        total_pnl = ce_result['pnl'] + pe_result['pnl']
        combined_return = ((ce_result['pnl'] + pe_result['pnl']) / 
                          (ce_result['entry_price'] + pe_result['entry_price'])) * 100
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="🎯 Total P&L",
                value=f"₹{total_pnl:.2f}",
                delta=f"{combined_return:.2f}%"
            )
        
        with col2:
            st.metric(
                label="💰 Total Premium Collected",
                value=f"₹{ce_result['entry_price'] + pe_result['entry_price']:.2f}"
            )

def main():
    # Title and description
    st.title("📈 Nifty ATM Option Sell Backtest")
    st.markdown("""
    **Strategy**: Sell ATM CE and PE options at 09:17, with 80% stop loss
    
    🎯 **Rules**:
    - Entry: 09:17 AM (sell both CE and PE)
    - Stop Loss: 80% above entry price
    - Exit: 15:15 PM (if SL not hit)
    """)
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Upload CSV Files", "Fetch from API"],
            help="Choose to upload CSV files or fetch data from Breeze API"
        )
        
        if data_source == "Fetch from API":
            symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
            
            # Calculate current week's Thursday for expiry
            today = datetime.today()
            days_ahead = 3 - today.weekday()  # Thursday is weekday 3
            if days_ahead <= 0:  # If today is Friday, Saturday, or Sunday
                days_ahead += 7  # Get next week's Thursday
            current_thursday = today + timedelta(days=days_ahead)
            
            expiry_date = st.date_input("Expiry Date", current_thursday)
            strike_price = st.number_input("ATM Strike Price", value=23000, step=50)
            from_date = st.date_input("From Date", datetime.today() - timedelta(days=1))
            to_date = st.date_input("To Date", datetime.today())
            interval = st.selectbox("Interval", ["1minute", "5minute", "15minute"])
    
    # Initialize backtester
    backtester = OptionsBacktester()
    
    # Data loading section
    df_ce, df_pe = None, None
    
    if data_source == "Upload CSV Files":
        st.header("📁 Upload Data Files")
        col1, col2 = st.columns(2)
        
        with col1:
            ce_file = st.file_uploader("Upload CE Data (CSV)", type=["csv"], key="ce")
            if ce_file:
                df_ce = pd.read_csv(ce_file)
                st.success(f"CE data loaded: {len(df_ce)} rows")
        
        with col2:
            pe_file = st.file_uploader("Upload PE Data (CSV)", type=["csv"], key="pe")
            if pe_file:
                df_pe = pd.read_csv(pe_file)
                st.success(f"PE data loaded: {len(df_pe)} rows")
    
    else:  # Fetch from API
        st.header("🔌 Fetch Data from API")
        
        if st.button("Fetch Data", type="primary"):
            breeze = load_api_connection()
            
            if breeze:
                with st.spinner("Fetching option data..."):
                    # Fetch CE data
                    df_ce = fetch_option_data(
                        breeze, symbol, strike_price, "call", 
                        pd.to_datetime(expiry_date), 
                        pd.to_datetime(from_date), 
                        pd.to_datetime(to_date), 
                        interval
                    )
                    
                    # Fetch PE data
                    df_pe = fetch_option_data(
                        breeze, symbol, strike_price, "put", 
                        pd.to_datetime(expiry_date), 
                        pd.to_datetime(from_date), 
                        pd.to_datetime(to_date), 
                        interval
                    )
                    
                    # Save to CSV for future use
                    if df_ce is not None:
                        df_ce.to_csv("ce_data.csv", index=False)
                        st.success("CE data saved to ce_data.csv")
                    
                    if df_pe is not None:
                        df_pe.to_csv("pe_data.csv", index=False)
                        st.success("PE data saved to pe_data.csv")
        
        # Load existing CSV if available
        if os.path.exists("ce_data.csv") and os.path.exists("pe_data.csv"):
            if st.button("Load Existing CSV Data"):
                df_ce = pd.read_csv("ce_data.csv")
                df_pe = pd.read_csv("pe_data.csv")
                st.success("Loaded existing CSV data")
    
    # Run backtest if data is available
    if df_ce is not None and df_pe is not None:
        st.header("🚀 Running Backtest")
        
        # Validate and preprocess data
        if backtester.validate_data(df_ce, "CE") and backtester.validate_data(df_pe, "PE"):
            df_ce = backtester.preprocess_data(df_ce)
            df_pe = backtester.preprocess_data(df_pe)
            
            # Execute backtest
            ce_result, pe_result = backtester.run_backtest(df_ce, df_pe)
            
            if ce_result or pe_result:
                # Display results
                display_results(ce_result, pe_result)
                
                # Create and display interactive plot
                if ce_result and pe_result:
                    st.header("📈 Interactive Chart")
                    fig = create_interactive_plot(df_ce, df_pe, ce_result, pe_result)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed results
                with st.expander("📋 Detailed Results"):
                    if ce_result:
                        st.subheader("CE Option Details")
                        st.json(ce_result)
                    
                    if pe_result:
                        st.subheader("PE Option Details")
                        st.json(pe_result)
            else:
                st.error("Backtest failed. Please check your data and parameters.")
    
    else:
        st.info("Please load or fetch option data to run the backtest.")

if __name__ == "__main__":
    main()
