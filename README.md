# Algo-Trading System

This project is an automated algorithmic trading system that fetches stock data, applies a trading strategy based on technical indicators and a machine learning model, and logs the trading signals to Google Sheets.

## Features

- **Data Ingestion:** Fetches daily stock data for NIFTY 50 stocks using the Yahoo Finance API.
- **Trading Strategy:** Generates buy/sell signals based on a combination of:
  - **Relative Strength Index (RSI):** To identify overbought or oversold conditions.
  - **Moving Average Crossover (20-DMA & 50-DMA):** To confirm trends.
- **ML-Powered Predictions:** Integrates a pre-trained machine learning model to predict next-day price movements, enhancing the trading signals.
- **Google Sheets Logging:** Automatically logs all trading signals, including ticker, date, closing price, and key indicators, to a designated Google Sheet for easy tracking and analysis.
- **Automation:** The entire workflow is orchestrated by a main script that can be scheduled to run at regular intervals.

## How It Works

1. **Data Fetching:** The system downloads the latest stock data for a predefined list of tickers.
2. **Signal Generation:** For each stock, it calculates the RSI and moving averages.
3. **ML Prediction:** It uses a saved machine learning model to predict whether the stock price will rise or fall.
4. **Trade Decision:** A final buy, sell, or hold signal is generated based on a combination of the technical indicators and the ML prediction.
5. **Logging:** The signal and relevant data points are appended to a Google Sheet.

## Usage

1. **Prerequisites:**

   - Python 3.12
   - Pip for package management

2. **Installation:**

   - Clone the repository.
   - Install the required packages:
     ```
     uv pip install -r requirements.txt
     ```

3. **Configuration:**

   - **Google Sheets API:**
     - Follow the `gspread` documentation to set up a service account and get your `credentials.json` file.
     - Share your Google Sheet with the client email from the credentials file.
     - Update the `sheet_id` in `src/sheets_integration.py` with your Google Sheet's ID.

4. **Running the System:**
   - Execute the main script to run the full workflow:
     ```
     python main.py
     ```

## Project Structure

```
ALGO-BOT
├── data/
│   └── (Stock data CSVs will be stored here)
├── models/
│   └── best_model.pkl
├── notebooks/
│   ├── ml_training.ipynb
│   └── trading_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── signals.py
│   ├── strategy.py
│   ├── sheets_integration.py
│   └── algo_automation.py
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

- **`data/`**: Stores the downloaded stock data.
- **`models/`**: Contains the trained machine learning model.
- **`notebooks/`**: Jupyter notebooks for model training and analysis.
- **`src/`**: Main source code for the application.
- **`main.py`**: The entry point to run the trading system.
