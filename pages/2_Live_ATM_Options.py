import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
import logging
import time
import sys
import os

# Add parent directory to path to import OptionDataFetcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import OptionDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("Live ATM Options Monitor")
    st.write("Monitor real-time ATM Call and Put options prices")
    
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
    
    # Add live ATM options chart button
    if st.button("Start Live Monitoring"):
        atm_strike = data_fetcher.get_atm_strike()
        if atm_strike:
            st.write(f"Current ATM Strike: {atm_strike}")
            
            # Create two columns for CE and PE current prices
            col1, col2 = st.columns(2)
            ce_price_display = col1.empty()
            pe_price_display = col2.empty()
            
            # Create placeholder for live chart
            chart_placeholder = st.empty()
            
            # Initialize data for both CE and PE
            ce_data = {'time': [], 'price': []}
            pe_data = {'time': [], 'price': []}
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[], y=[], name='CE', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=[], y=[], name='PE', line=dict(color='red')))
            
            fig.update_layout(
                title=f"{symbol} ATM {atm_strike} Live Options Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Update function for live data
            try:
                while True:
                    ce_quote, pe_quote = data_fetcher.get_live_atm_data()
                    if ce_quote and pe_quote:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        
                        # Update CE data
                        ce_price = float(ce_quote['ltp'])
                        ce_data['time'].append(current_time)
                        ce_data['price'].append(ce_price)
                        
                        # Update PE data
                        pe_price = float(pe_quote['ltp'])
                        pe_data['time'].append(current_time)
                        pe_data['price'].append(pe_price)
                        
                        # Update current prices display
                        ce_price_display.metric("CE Price", f"₹{ce_price:.2f}", 
                                             f"{ce_price - ce_data['price'][-2]:.2f}" if len(ce_data['price']) > 1 else None)
                        pe_price_display.metric("PE Price", f"₹{pe_price:.2f}", 
                                             f"{pe_price - pe_data['price'][-2]:.2f}" if len(pe_data['price']) > 1 else None)
                        
                        # Update traces
                        fig.data[0].x = ce_data['time'][-50:]  # Keep last 50 points
                        fig.data[0].y = ce_data['price'][-50:]
                        fig.data[1].x = pe_data['time'][-50:]
                        fig.data[1].y = pe_data['price'][-50:]
                        
                        # Update chart
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Wait for 1 second before next update
                        time.sleep(1)
            except Exception as e:
                st.error(f"Error updating live data: {e}")
        else:
            st.error("Could not determine ATM strike price. Please check market hours and connectivity.")

if __name__ == "__main__":
    main()
