from abc import ABC, abstractmethod
from typing import Union
from model.data_cleaning import DataLoaderStrategy, DataDivideStrategy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelBuilder(ABC):
    """
    Model builder class which builds the model
    """
    @abstractmethod
    def build_model(self, df: pd.DataFrame) -> Union[RandomForestRegressor, GradientBoostingRegressor]:
        pass

class RFRegressorBuilder(ModelBuilder):
    """
    Random Forest Regressor model builder
    """
    def build_model(self, df: pd.DataFrame) -> RandomForestRegressor:
        """
        Builds the Random Forest Regressor model
        """
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        rf = RandomForestRegressor(random_state=101)
        rf.fit(X_train, y_train)
        return rf
    
class XGRegressorBuilder(ModelBuilder):
    """
    XGBoost Regressor model builder class
    """
    def build_model(self, df: pd.DataFrame) -> XGBRegressor:
        """
        Builds the XGBoost Regressor model
        """
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        gbr = GradientBoostingRegressor(random_state=101)
        gbr.fit(X_train, y_train)
        return xgb
    
df = DataLoaderStrategy().handle_data("../data/Final.csv")
rfr = RFRegressorBuilder().build_model(df)
xgb = XGRegressorBuilder().build_model(df)

print(type(rfr))
print(type(xgb))