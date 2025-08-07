import yfinance as yf
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the list of NIFTY 50 stocks to fetch
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS"
]

def fetch_stock_data(tickers, start_date, end_date, data_dir="data"):
    """
    Fetch stock data for the given tickers and save to CSV files."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for ticker in tickers:
        try:
            logging.info(f"Fetching data for {ticker}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                file_path = os.path.join(data_dir, f"{ticker}.csv")
                stock_data.to_csv(file_path)
                logging.info(f"Data for {ticker} saved to {file_path}")
            else:
                logging.warning(f"No data found for {ticker}")
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")

if __name__ == "__main__":
    # Set the start and end dates for the data
    start_date = "2024-04-01"
    end_date = "2025-03-31"

    # Fetch data for the NIFTY 50 stocks
    fetch_stock_data(NIFTY_50_STOCKS, start_date, end_date)