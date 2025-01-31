<<<<<<< HEAD
# Option Chain Charts

A Streamlit web application for visualizing option chain data using candlestick charts.

## Features

- Interactive candlestick charts with volume and open interest
- Support for both 1-minute and 5-minute intervals
- Filter by:
  - Symbol (NIFTY/BANKNIFTY)
  - Strike Price
  - Option Type (CE/PE)
  - Expiry Date
  - Date Range
- Dark theme for better visualization
- Data table view

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Breeze API credentials:
```
API_KEY=your_api_key
API_SECRET=your_api_secret
```

3. Make sure you have a valid `session_token.json` file with your Breeze session token.

## Running the App

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

## Usage

1. Select your desired parameters in the sidebar:
   - Choose the symbol (NIFTY/BANKNIFTY)
   - Set the date range
   - Enter the strike price
   - Select option type (CE/PE)
   - Choose the expiry date
   - Select the interval (1min/5min)

2. Click "Fetch Data" to generate the chart

3. The chart will display:
   - Candlesticks for price action
   - Volume bars (right axis)
   - Open Interest line (far right axis)

4. Below the chart, you'll find a data table with all the fetched data
=======
# options_chart_hist
To get 1min and 5min options historical data
>>>>>>> 38a9a1e991baa755718dd4d45c59c5726ee0409f
