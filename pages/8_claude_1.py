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

def fetch_spot_data(breeze, symbol: str, from_date: datetime, to_date: datetime, 
                   interval: str) -> Optional[pd.DataFrame]:
    """Fetch spot/cash data from Breeze API"""
    try:
        # Map symbol to exchange code
        exchange_code = "NSE" if symbol in ["NIFTY", "BANKNIFTY"] else "NSE"
        stock_code = "NIFTY 50" if symbol == "NIFTY" else "NIFTY BANK"
        
        response = breeze.get_historical_data_v2(
            interval=interval,
            from_date=from_date.strftime("%Y-%m-%dT07:00:00.000Z"),
            to_date=(to_date + timedelta(days=1)).strftime("%Y-%m-%dT07:00:00.000Z"),
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="cash"
        )
        
        if "Success" not in response or not response["Success"]:
            st.error(f"No spot data returned for {symbol}")
            return None
        
        df = pd.DataFrame(response["Success"])
        return df
    
    except Exception as e:
        st.error(f"Error fetching spot data: {e}")
        return None

def calculate_atm_strike(spot_price: float, symbol: str) -> int:
    """Calculate ATM strike price from spot price"""
    if symbol == "NIFTY":
        # NIFTY strikes are in multiples of 50
        strike_multiple = 50
    elif symbol == "BANKNIFTY":
        # BANKNIFTY strikes are in multiples of 100
        strike_multiple = 100
    else:
        strike_multiple = 50  # Default
    
    # Round to nearest strike multiple
    atm_strike = round(spot_price / strike_multiple) * strike_multiple
    return int(atm_strike)

def get_nearby_strikes(atm_strike: int, symbol: str, count: int = 2) -> list:
    """Get nearby strike prices around ATM"""
    if symbol == "NIFTY":
        strike_gap = 50
    elif symbol == "BANKNIFTY":
        strike_gap = 100
    else:
        strike_gap = 50
    
    strikes = []
    for i in range(-count, count + 1):
        strike = atm_strike + (i * strike_gap)
        strikes.append(strike)
    
    return strikes
    """Get spot price at specific time (e.g., 09:17)"""
    if df_spot is None or df_spot.empty:
        return None
    
    # Find closest time to target
    df_spot['datetime'] = pd.to_datetime(df_spot['datetime'])
    df_spot = df_spot.sort_values('datetime')
    
    target_rows = df_spot[df_spot['datetime'].dt.time == target_time]
    if not target_rows.empty:
        return float(target_rows.iloc[0]['close'])
    
    # If exact time not found, find closest time after target
    target_datetime = datetime.combine(df_spot.iloc[0]['datetime'].date(), target_time)
    closest_row = df_spot[df_spot['datetime'] >= target_datetime]
    
    if not closest_row.empty:
        return float(closest_row.iloc[0]['close'])
    
    return None
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

def display_strategy_summary(symbol: str, strike_price: int, expiry_date: datetime, 
                           is_auto_calculated: bool = False, spot_price: float = None):
    """Display a summary of the strategy parameters"""
    st.markdown("### 📋 Strategy Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Symbol", symbol)
        st.metric("Strike Price", f"₹{strike_price}")
    
    with col2:
        st.metric("Expiry Date", expiry_date.strftime("%d-%b-%Y"))
        if is_auto_calculated and spot_price:
            st.metric("Spot Price @ 09:17", f"₹{spot_price:.2f}")
    
    with col3:
        st.metric("Entry Time", "09:17 AM")
        st.metric("Stop Loss", "80% above entry")
    
    if is_auto_calculated:
        st.success("✅ Strike price auto-calculated from spot price")
    else:
        st.info("🔧 Manual strike price selected")
    
    st.markdown("---")
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
    
    🚀 **New Features**:
    - ✅ Auto-calculate ATM strike from spot price
    - ✅ View nearby strikes (±2 levels)
    - ✅ Enhanced visualization with Plotly
    - ✅ Comprehensive trade analytics
    """)
    
    # Add a tip box
    st.info("""
    💡 **Pro Tip**: Use "Auto Calculate ATM" to automatically determine the strike price 
    from Nifty spot price at 09:17. This ensures true ATM trading!
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
            
            # Strike price selection method
            strike_method = st.radio(
                "Strike Price Method",
                ["Manual Input", "Auto Calculate ATM"],
                help="Choose to manually enter strike or auto-calculate from spot price"
            )
            
            if strike_method == "Manual Input":
                strike_price = st.number_input("ATM Strike Price", value=23000, step=50)
            else:
                strike_price = None  # Will be calculated automatically
                st.info("Strike price will be auto-calculated from spot price at 09:17")
                
                # Option to show nearby strikes
                show_nearby = st.checkbox("Show nearby strike options", 
                                        help="Display ATM ±2 strikes for selection")
                if show_nearby:
                    st.warning("💡 Nearby strikes will be shown after fetching spot data")
            
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
                with st.spinner("Fetching data..."):
                    # Step 1: Fetch spot data if auto-calculating strike
                    calculated_strike = None
                    if strike_price is None:  # Auto calculate
                        st.info("📊 Fetching spot data to calculate ATM strike...")
                        df_spot = fetch_spot_data(
                            breeze, symbol,
                            pd.to_datetime(from_date), 
                            pd.to_datetime(to_date), 
                            interval
                        )
                        
                        if df_spot is not None:
                            spot_price = get_spot_price_at_time(df_spot, backtester.entry_time)
                            if spot_price:
                                calculated_strike = calculate_atm_strike(spot_price, symbol)
                                st.success(f"✅ Spot price at 09:17: ₹{spot_price:.2f}")
                                st.success(f"🎯 Calculated ATM strike: {calculated_strike}")
                                
                                # Show nearby strikes if requested
                                if 'show_nearby' in locals() and show_nearby:
                                    nearby_strikes = get_nearby_strikes(calculated_strike, symbol)
                                    st.write("📊 **Available Strikes:**")
                                    
                                    strike_cols = st.columns(len(nearby_strikes))
                                    for i, strike in enumerate(nearby_strikes):
                                        with strike_cols[i]:
                                            if strike == calculated_strike:
                                                st.success(f"**{strike}** (ATM)")
                                            else:
                                                otm_itm = "OTM" if abs(strike - calculated_strike) > 0 else "ATM"
                                                st.write(f"{strike} ({otm_itm})")
                                    
                                    # Allow user to select different strike
                                    selected_strike = st.selectbox(
                                        "Choose Strike Price",
                                        nearby_strikes,
                                        index=nearby_strikes.index(calculated_strike),
                                        help="Select the strike price for backtesting"
                                    )
                                    strike_price = selected_strike
                                    if selected_strike != calculated_strike:
                                        st.info(f"Selected strike {selected_strike} instead of ATM {calculated_strike}")
                                else:
                                    strike_price = calculated_strike
                            else:
                                st.error("Could not find spot price at 09:17. Please use manual strike input.")
                                st.stop()
                        else:
                            st.error("Failed to fetch spot data. Please use manual strike input.")
                            st.stop()
                    
                    # Step 2: Fetch option data with determined strike price
                    st.info(f"📈 Fetching option data for strike {strike_price}...")
                    
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
                        filename = f"ce_data_{symbol}_{strike_price}_{expiry_date.strftime('%Y%m%d')}.csv"
                        df_ce.to_csv(filename, index=False)
                        st.success(f"CE data saved to {filename}")
                    
                    if df_pe is not None:
                        filename = f"pe_data_{symbol}_{strike_price}_{expiry_date.strftime('%Y%m%d')}.csv"
                        df_pe.to_csv(filename, index=False)
                        st.success(f"PE data saved to {filename}")
                    
                    # Display strike info
                    if calculated_strike:
                        st.info(f"🎯 **Auto-calculated ATM Strike**: {calculated_strike} (from spot price)")
                    else:
                        st.info(f"🎯 **Manual Strike**: {strike_price}")
        
        # Load existing CSV if available
        csv_pattern = f"*{symbol}*{expiry_date.strftime('%Y%m%d')}*.csv"
        import glob
        ce_files = glob.glob(f"ce_data_{csv_pattern}")
        pe_files = glob.glob(f"pe_data_{csv_pattern}")
        
        if ce_files and pe_files:
            if st.button("Load Most Recent CSV Data"):
                # Get most recent files
                ce_file = max(ce_files, key=os.path.getctime)
                pe_file = max(pe_files, key=os.path.getctime)
                
                df_ce = pd.read_csv(ce_file)
                df_pe = pd.read_csv(pe_file)
                
                # Extract strike from filename
                import re
                strike_match = re.search(r'_(\d+)_', ce_file)
                if strike_match:
                    loaded_strike = int(strike_match.group(1))
                    st.success(f"Loaded data for strike {loaded_strike} from: {ce_file}")
                else:
                    st.success(f"Loaded existing CSV data")
    
    # Display current parameters
    if 'strike_price' in locals() and strike_price:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Current Parameters")
        st.sidebar.write(f"**Symbol**: {symbol if 'symbol' in locals() else 'Not set'}")
        st.sidebar.write(f"**Strike Price**: {strike_price}")
        st.sidebar.write(f"**Expiry**: {expiry_date if 'expiry_date' in locals() else 'Not set'}")
        if 'calculated_strike' in locals() and calculated_strike:
            st.sidebar.success("✅ Auto-calculated ATM")
        else:
            st.sidebar.info("🔧 Manual strike")
    
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
                # Display strategy summary
                if 'calculated_strike' in locals() and calculated_strike:
                    spot_at_entry = get_spot_price_at_time(
                        fetch_spot_data(breeze, symbol, pd.to_datetime(from_date), 
                                      pd.to_datetime(to_date), interval) if 'breeze' in locals() else None, 
                        backtester.entry_time
                    ) if 'symbol' in locals() else None
                    display_strategy_summary(
                        symbol if 'symbol' in locals() else "N/A", 
                        calculated_strike, 
                        pd.to_datetime(expiry_date) if 'expiry_date' in locals() else datetime.now(),
                        is_auto_calculated=True, 
                        spot_price=spot_at_entry
                    )
                elif 'symbol' in locals() and 'strike_price' in locals():
                    display_strategy_summary(
                        symbol, 
                        strike_price, 
                        pd.to_datetime(expiry_date) if 'expiry_date' in locals() else datetime.now(),
                        is_auto_calculated=False
                    )
                
                # Display results
                display_results(ce_result, pe_result)
                
                # Display ATM calculation info if applicable
                if 'calculated_strike' in locals() and calculated_strike:
                    st.info(f"ℹ️ **ATM Strike Calculation**: Used spot price at 09:17 to determine strike {calculated_strike}")
                
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
