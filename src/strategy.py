import pandas as pd
from src.signals import generate_signals

def backtest_strategy(data):
    """Backtest the trading strategy."""
    data = generate_signals(data)
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    positions['stock'] = data['signal'].diff()
    
    portfolio = pd.DataFrame(index=data.index).fillna(0.0)
    portfolio['holdings'] = (positions['stock'] * data['Close']).cumsum()
    portfolio['cash'] = 100000 - (positions['stock'] * data['Close']).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    
    return portfolio
