
import pandas as pd

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(data, short_window=20, long_window=50):
    """Calculate short and long moving averages."""
    data['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

def generate_signals(data):
    """Generate trading signals based on the strategy."""
    data['rsi'] = calculate_rsi(data)
    data = calculate_moving_averages(data)
    
    data['signal'] = 0
    data.loc[(data['rsi'] < 30) & (data['short_mavg'] > data['long_mavg']), 'signal'] = 1
    
    return data
