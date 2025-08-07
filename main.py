from src.algo_automation import AlgoTradingSystem
import os

# --- Configuration ---
NIFTY_50_STOCKS = ['BAJFINANCE.NS', 'BHARTIARTL.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INFY.NS', 'KOTAKBANK.NS', 'RELIANCE.NS', 'SBIN.NS', 'TCS.NS'] # Example stocks

def main():
    system = AlgoTradingSystem(NIFTY_50_STOCKS)
    system.run_full_workflow()

if __name__ == "__main__":
    main()