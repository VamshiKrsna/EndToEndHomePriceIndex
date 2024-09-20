from datetime import datetime
import os
import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class for data preprocessing and data splitting."""

    @abstractmethod
    def handle_data(self,data:pd.DataFrame):
        pass

class DataPreprocessingStrategy(DataStrategy):
    """
    Merge Data Strategy which merges the data from individual csv files initially.
    """
    def handle_data(self, data: pd.DataFrame = None)->pd.DataFrame:
        """
        clean, merge all csv files
        """
        try:
            cwd = os.getcwd()
            data_dir = os.path.join(cwd, "..", "data")
            data_dir = os.path.normpath(data_dir)
            cpi = pd.read_csv(os.path.join(data_dir, "CPIAUCSL.csv"))
            csus = pd.read_csv(os.path.join(data_dir, "CSUSHPISA.csv"))
            fedfunds = pd.read_csv(os.path.join(data_dir, "FEDFUNDS.csv"))
            gdp = pd.read_csv(os.path.join(data_dir, "GDB.csv"))
            houst = pd.read_csv(os.path.join(data_dir, "HOUST.csv"))
            mcoil = pd.read_csv(os.path.join(data_dir, "MCOILWTICO.csv"))
            meho = pd.read_csv(os.path.join(data_dir, "MEHOINUSA672N.csv"))
            pop = pd.read_csv(os.path.join(data_dir, "POP.csv"))
            unrate = pd.read_csv(os.path.join(data_dir, "UNRATE.csv"))
            permit = pd.read_csv(os.path.join(data_dir, "PERMIT.csv"))
            # After prior research in "US Home Price.ipynb", following decisions had to be made : 
            csus = csus.dropna()
            csus.reset_index(drop=True, inplace=True)
            # Updating all dfs in required date range only : 
            cpi = cpi[(cpi["date"] >= "1987-01-01") & (cpi["date"] <= "2024-04-01")]
            csus = csus[(csus["date"] >= "1987-01-01") & (csus["date"] <= "2024-04-01")]
            fedfunds = fedfunds[(fedfunds["date"] >= "1987-01-01") & (fedfunds["date"] <= "2024-04-01")]
            gdp = gdp[(gdp["date"] >= "1987-01-01") & (gdp["date"] <= "2024-04-01")]
            houst = houst[(houst["date"] >= "1987-01-01") & (houst["date"] <= "2024-04-01")]
            mcoil = mcoil[(mcoil["date"] >= "1987-01-01") & (mcoil["date"] <= "2024-04-01")]
            meho = meho[(meho["date"] >= "1987-01-01") & (meho["date"] <= "2024-04-01")]
            pop = pop[(pop["date"] >= "1987-01-01") & (pop["date"] <= "2024-04-01")]
            unrate = unrate[(unrate["date"] >= "1987-01-01") & (unrate["date"] <= "2024-04-01")]
            # making sure all data have datetime format : 
            cpi["date"] = pd.to_datetime(cpi["date"], format="%Y-%m-%d")
            csus["date"] = pd.to_datetime(csus["date"], format="%Y-%m-%d")
            fedfunds["date"] = pd.to_datetime(fedfunds["date"], format="%Y-%m-%d")
            gdp["date"] = pd.to_datetime(gdp["date"], format="%Y-%m-%d")
            houst["date"] = pd.to_datetime(houst["date"], format="%Y-%m-%d")
            mcoil["date"] = pd.to_datetime(mcoil["date"], format="%Y-%m-%d")
            meho["date"] = pd.to_datetime(meho["date"], format="%Y-%m-%d")
            pop["date"] = pd.to_datetime(pop["date"], format="%Y-%m-%d")
            unrate["date"] = pd.to_datetime(unrate["date"], format="%Y-%m-%d")
            # fixing gdp dataframe:
            df = gdp.copy()
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="MS")
            df = df.reindex(date_range)
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)
            df["value"] = df["value"].interpolate(method="linear")
            new_data = pd.DataFrame({
                "value": [29234.51, 29365.95, 29365.95],
                "date": ["2024-02-01","2024-03-01","2024-04-01"]
            }, index=[445,446,447])
            new_data["date"] = pd.to_datetime(new_data["date"], format="%Y-%m-%d")
            df = pd.concat([df, new_data])
            gdp = df.copy()
            # Fixing MEHO :
            temp = meho.copy()
            temp["date"] = pd.to_datetime(meho["date"])
            temp.reset_index(drop=True, inplace=True)
            temp.set_index("date", inplace=True)
            temp_resampled = temp.resample("MS").mean()
            temp_interpolated = temp_resampled.interpolate(method="linear")
            last_date = '2024-04-01'
            last_date = pd.to_datetime(last_date)
            extended_index = pd.date_range(start=temp_interpolated.index.min(), end=last_date, freq='MS')
            df_extended = temp_interpolated.reindex(extended_index)
            df_interpolated = df_extended.interpolate(method='linear')
            df_interpolated.rename(columns={'index': 'date'}, inplace=True)
            df_interpolated = df_interpolated.iloc[:, 1:] 
            df_interpolated = df_interpolated[:-1]
            meho = df_interpolated.copy()
            new_data = {'date': pd.to_datetime('2024-04-01'), 'value': 29365.95}
            new_data_df = pd.DataFrame([new_data])
            meho = pd.concat([meho, new_data_df], ignore_index=True)
            meho = meho.sort_values(by='date')
            # Handling Permit : 
            permit.loc[len(permit)] = ['2024-04-01', 1485.0]
            permit['date'] = pd.to_datetime(permit['date'])
            permit = permit.copy()
            # Building Final Data Frame : 
            cpi.rename(columns={"value": "CPI"}, inplace=True)
            csus.rename(columns={"value": "Home Price Index"}, inplace=True)
            fedfunds.rename(columns={"value": "Federal Funds Rate"}, inplace=True)
            gdp.rename(columns={"value": "GDP"}, inplace=True)
            houst.rename(columns={"value": "Housing Starts"}, inplace=True)
            mcoil.rename(columns={"value": "Crude Oil Prices"}, inplace=True)
            meho.rename(columns={"value": "Median Household Income"}, inplace=True)
            pop.rename(columns={"value": "Population"}, inplace=True)
            unrate.rename(columns={"value": "Unemployment Rate"}, inplace=True)
            permit.rename(columns={"value": "Building Permits"}, inplace=True)
            for df in [cpi, csus, fedfunds, gdp, houst, mcoil, meho, pop, unrate, permit]:
                df['date'] = pd.to_datetime(df['date'])

            cpi.set_index('date', inplace=True)
            csus.set_index('date', inplace=True)
            fedfunds.set_index('date', inplace=True)
            gdp.set_index('date', inplace=True)
            houst.set_index('date', inplace=True)
            mcoil.set_index('date', inplace=True)
            meho.set_index('date', inplace=True)
            pop.set_index('date', inplace=True)
            unrate.set_index('date', inplace=True)
            permit.set_index('date', inplace=True)

            merged_df = pd.concat([cpi, csus, fedfunds, gdp, houst, mcoil, meho, pop, unrate, permit], axis=1)
            merged_df.reset_index(inplace=True)
            
            merged_df.to_csv("../data/final_df.csv")

            return merged_df
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)

df = DataPreprocessingStrategy().handle_data()
print(df)