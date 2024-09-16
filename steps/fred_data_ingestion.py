import os
import logging
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from zenml import step

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

class FredDataIngestion:
    """Class responsible for fetching data from the FRED API and saving it as CSV."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize the Fred API with the provided API key."""
        self.fred = Fred(api_key=api_key)
        self.data_dir = '../data'
        self.series_list = [
            'CPIAUCSL',  # Consumer Price Index for All Urban Consumers: Seasonally Adjusted (CPI-U)
            'CSUSHPISA',  # S&P Case-Shiller US National Home Price Index
            'FEDFUNDS',  # Effective Federal Funds Rate
            'GDB',  # Gross Domestic Product
            'HOUST',  # Housing Starts
            'MCOILWTICO',  # Crude Oil Prices (West Texas)
            'MEHOINUSA672N',  # Median Household Income
            'MORTGAGE15US',  # 15 Years Fixed rate mortgage average
            'MORTGAGE30US',  # 30 Years Fixed rate mortgage average
            'POP',  # Population in the US
            'UNRATE'  # Unemployment Rate
        ]
        
        # Create data directory if not exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_series(self, series: str) -> pd.DataFrame:
        """Fetch a series from FRED API and return a DataFrame."""
        data = self.fred.get_series(series)
        df = data.reset_index()
        df.columns = ['date', 'value']
        return df

    def save_to_csv(self, df: pd.DataFrame, series: str) -> None:
        """Save the DataFrame as a CSV file."""
        file_path = os.path.join(self.data_dir, f'{series}.csv')
        df.to_csv(file_path, index=False)
        logging.info(f'Saved {series} to {file_path}')
    
    def ingest(self) -> None:
        """Ingest data for each series in the list and save it as CSV."""
        for series in self.series_list:
            try:
                df = self.fetch_series(series)
                self.save_to_csv(df, series)
                logging.info(f'Data sample for {series}:\n{df.head()}')
            except Exception as e:
                logging.error(f"Error fetching or saving data for {series}: {e}")
                raise e


@step
def ingest_fred_data() -> None:
    """ZenML step to ingest FRED data."""
    try:
        fred_data_ingestor = FredDataIngestion(api_key=FRED_API_KEY)
        fred_data_ingestor.ingest()
    except Exception as e:
        logging.error(f"Failed to ingest data: {e}")
        raise e
    
# ingest_fred_data() # This script works so well!