import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(root))

class DataLoaderStrategy:
    """Data Loading Class"""
    def handle_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy:
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def split_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            data = data.drop("date",axis = 1)
            X = data.drop("Home Price Index", axis=1)
            y = data["Home Price Index"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
# df = DataLoaderStrategy().handle_data("../data/Final.csv")
# print(df)

# X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
# print(X_train, X_test, y_train, y_test)