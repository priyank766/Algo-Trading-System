import pandas as pd
import os
import numpy as np
from datetime import datetime
import pickle
import logging
from .data_ingestion import fetch_stock_data
from .signals import calculate_rsi, calculate_moving_averages
from .sheets_integration import log_trades_to_sheet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlgoTradingSystem:
    def __init__(self, stock_symbols, data_path='data/', models_path='models/'):
        self.stock_symbols = stock_symbols
        self.data_path = data_path
        self.models_path = models_path
        self.ml_model = self._load_ml_model()

    def _load_ml_model(self):
        model_path = os.path.join(self.models_path, 'best_model.pkl')
        if os.path.exists(model_path):
            logging.info(f"Loading ML model from {model_path}")
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            logging.warning(f"ML model not found at {model_path}.")
            return None

    def run_data_ingestion_with_dates(self, start_date, end_date):
        logging.info(f"Running data ingestion from {start_date} to {end_date}...")
        fetch_stock_data(self.stock_symbols, start_date, end_date, self.data_path)

    def run_strategy_and_ml(self):
        logging.info("Running trading strategy and ML predictions...")
        all_processed_data = {}
        column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        for symbol in self.stock_symbols:
            file_path = os.path.join(self.data_path, f'{symbol}.csv')
            if not os.path.exists(file_path):
                logging.warning(f"Skipping {symbol}: Data file not found at {file_path}")
                continue

            try:
                df = pd.read_csv(file_path, header=0)
                if 'Date' not in df.columns:
                    df = pd.read_csv(file_path, skiprows=3, header=None, names=column_names)
            except Exception as e:
                logging.error(f"Could not process {symbol}.csv. Error: {e}. Skipping file.")
                continue

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            df['RSI'] = calculate_rsi(df.copy())
            df = calculate_moving_averages(df.copy())
            df['20_DMA'] = df['short_mavg']
            df['50_DMA'] = df['long_mavg']

            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            df['Volume_Change'] = df['Volume'].diff()

            df = df.dropna()

            if not df.empty and self.ml_model:
                features = ['RSI', 'MACD', 'Volume_Change']
                X_latest = df[features].iloc[-1:].copy()

                try:
                    ml_prediction = self.ml_model.predict(X_latest)[0]
                    ml_proba = self.ml_model.predict_proba(X_latest)[0][1]
                    logging.info(f"  {symbol}: ML Prediction: {ml_prediction} (Probability: {ml_proba:.2f})")
                    df['ML_Prediction'] = ml_prediction
                    df['ML_Probability'] = ml_proba
                except Exception as e:
                    logging.error(f"  Error during ML prediction for {symbol}: {e}")
                    df['ML_Prediction'] = np.nan
                    df['ML_Probability'] = np.nan
            else:
                df['ML_Prediction'] = np.nan
                df['ML_Probability'] = np.nan

            df['Trading_Signal'] = 'HOLD'
            if not df.empty:
                latest_rsi = df['RSI'].iloc[-1]
                latest_short_mavg = df['20_DMA'].iloc[-1]
                latest_long_mavg = df['50_DMA'].iloc[-1]
                latest_ml_pred = df['ML_Prediction'].iloc[-1]

                if latest_rsi < 30 and latest_short_mavg > latest_long_mavg and latest_ml_pred == 1:
                    df.loc[df.index[-1], 'Trading_Signal'] = 'BUY'
                elif latest_rsi > 70 and latest_short_mavg < latest_long_mavg and latest_ml_pred == 0:
                    df.loc[df.index[-1], 'Trading_Signal'] = 'SELL'

            all_processed_data[symbol] = df

        logging.info("Strategy and ML prediction complete.")
        return all_processed_data

    def log_to_google_sheets(self, processed_data):
        logging.info("Preparing data for Google Sheets...")
        
        latest_signals = []
        for symbol, df in processed_data.items():
            if not df.empty:
                latest_row = df.iloc[-1].copy()
                latest_signals.append({
                    'Ticker': symbol,
                    'Date': latest_row['Date'].strftime('%Y-%m-%d'),
                    'Close': latest_row['Close'],
                    'RSI': latest_row['RSI'],
                    '20_DMA': latest_row['20_DMA'],
                    '50_DMA': latest_row['50_DMA'],
                    'ML_Prediction': latest_row['ML_Prediction'],
                    'Trading_Signal': latest_row['Trading_Signal']
                })
        
        if latest_signals:
            sheets_df = pd.DataFrame(latest_signals)
            logging.info("Logging to Google Sheets...")
            log_trades_to_sheet(sheets_df)
        else:
            logging.warning("No data to log to Google Sheets.")

    def run_full_workflow(self, start_date, end_date):
        logging.info(f"--- Running Algo Trading Workflow ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
        self.run_data_ingestion_with_dates(start_date, end_date)
        processed_data = self.run_strategy_and_ml()
        self.log_to_google_sheets(processed_data)
        logging.info("--- Workflow complete ---")
