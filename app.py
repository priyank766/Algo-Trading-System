import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.algo_automation import AlgoTradingSystem
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

st.set_page_config(layout="wide")

st.title("Algo-Trading System")

st.markdown("---")

# --- Helper Functions for Model Evaluation ---

def create_features(df):
    df = df.sort_values(by=['Stock', 'Date'])
    delta = df.groupby('Stock')['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    exp2 = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = exp1 - exp2
    df['Volume_Change'] = df.groupby('Stock')['Volume'].transform(lambda x: x.diff())
    df['Target'] = df.groupby('Stock')['Close'].transform(lambda x: (x.shift(-1) > x).astype(int))
    return df.dropna()

@st.cache_data
def load_and_evaluate_model():
    # Load all historical data
    data_path = 'data/'
    stock_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    all_data_frames = []
    column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    for stock_file in stock_files:
        try:
            df = pd.read_csv(os.path.join(data_path, stock_file), header=0)
            if 'Date' not in df.columns:
                df = pd.read_csv(os.path.join(data_path, stock_file), skiprows=3, header=None, names=column_names)
        except Exception:
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        df['Stock'] = stock_file.replace('.csv', '')
        all_data_frames.append(df)

    combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    # Feature Engineering and Splitting
    processed_df = create_features(combined_df)
    features = ['RSI', 'MACD', 'Volume_Change']
    X = processed_df[features]
    y = processed_df['Target']
    train_size = int(len(processed_df) * 0.8)
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # Load Model and Evaluate
    model_path = 'models/best_model.pkl'
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    performance = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    return performance

# --- Streamlit UI ---

st.sidebar.header("Configuration")

DATA_DIR = 'data/'
if os.path.exists(DATA_DIR):
    available_stock_files = [f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    default_stocks = [s for s in ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'] if s in available_stock_files]
    if not default_stocks and available_stock_files:
        default_stocks = [available_stock_files[0]]
else:
    available_stock_files = []
    default_stocks = []

selected_stocks = st.sidebar.multiselect(
    "Select Stock Symbols",
    options=available_stock_files,
    default=default_stocks
)

today = datetime.now()
start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=today)

if start_date > end_date:
    st.sidebar.error("Error: End Date must be after Start Date.")
    st.stop()

st.markdown("---")

if st.sidebar.button("Run Algo-Trading System"):
    if not selected_stocks:
        st.error("Please select at least one stock symbol.")
    else:
        with st.spinner("Running the Algo-Trading System..."):
            system = AlgoTradingSystem(selected_stocks)
            system.run_data_ingestion_with_dates(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            processed_data = system.run_strategy_and_ml()
            
            # Log results to Google Sheets
            if processed_data:
                system.log_to_google_sheets(processed_data)

        st.header("Trading Results")
        if processed_data:
            for symbol, df in processed_data.items():
                st.subheader(f"Results for {symbol}")
                if not df.empty:
                    latest_row = df.iloc[-1]
                    cols = st.columns(4)
                    cols[0].metric("Latest Close", f"{latest_row['Close']:.2f}")
                    cols[1].metric("RSI", f"{latest_row['RSI']:.2f}")
                    cols[2].metric("Trading Signal", latest_row['Trading_Signal'])
                    cols[3].metric("ML Prediction", "Buy" if latest_row['ML_Prediction'] == 1 else "Sell")
                    
                    with st.expander("View Full Data"):
                        st.dataframe(df)
                else:
                    st.write("No data or signals generated for this stock.")
        else:
            st.warning("No trading results generated.")
        
        st.success("Algo-Trading System run complete!")

        st.markdown("---")
        st.header("Model Performance on Historical Data")
        with st.spinner("Evaluating model performance..."):
            performance = load_and_evaluate_model()
        
        if performance:
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{performance['Accuracy']:.2%}")
            cols[1].metric("F1-Score", f"{performance['F1-Score']:.4f}")
            cols[2].metric("Precision", f"{performance['Precision']:.4f}")
            cols[3].metric("ROC AUC", f"{performance['ROC AUC']:.4f}")
        else:
            st.error("Could not load or evaluate the model. Please ensure 'models/best_model.pkl' exists.")

st.markdown("---")
st.info("To run the system, select your desired stocks and date range, then click the 'Run' button in the sidebar.")