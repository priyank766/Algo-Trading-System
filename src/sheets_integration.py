import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets"
]

CREDS_FILE = 'credentials.json'

def get_gspread_client():
    """Authenticate with Google Sheets and return the client."""
    try:
        creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        logging.error(f"Failed to authenticate with Google Sheets: {e}")
        return None


sheet_id="1NYC0sqNt9v0KdTUvDcMFSf_sxmp8cQKxEUWHSzc5vhs"

def log_trades_to_sheet(data_df):
    """Appends a DataFrame to the first worksheet of a specified Google Sheet."""
    try:
        client = get_gspread_client()
        if client:
            spreadsheet = client.open_by_key(sheet_id)
            worksheet = spreadsheet.sheet1
            
            # Get all records to check if the sheet is empty
            existing_data = worksheet.get_all_records()
            
            # If the sheet is empty, write the header row first
            if not existing_data:
                worksheet.update([data_df.columns.values.tolist()])

            # Append the new data rows
            worksheet.append_rows(data_df.values.tolist())
            
            logging.info(f"Successfully appended data to the worksheet in sheet ID '{sheet_id}'")

    except Exception as e:
        logging.error(f"Error logging to Google Sheets: {e}")

if __name__ == '__main__':
   
    dummy_data = {
        'Ticker': ['RELIANCE.NS', 'TCS.NS'],
        'Signal': ['Buy', 'Sell'],
        'Price': [2800.50, 3500.75],
        'Timestamp': ['2024-05-20', '2024-05-21']
    }
    df = pd.DataFrame(dummy_data)

    log_trades_to_sheet(df)
