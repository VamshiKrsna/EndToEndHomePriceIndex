from abc import ABC, abstractmethod
from typing import Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor  # Ensure XGBoost is installed
import pandas as pd
class ModelBuilder(ABC):
    """
    Abstract base class for building models.
    """
    @abstractmethod
    def build_model(self, df: pd.DataFrame) -> Union[RandomForestRegressor, GradientBoostingRegressor, XGBRegressor]:
        pass

    @abstractmethod
    def evaluate_model(self, model, X_test, y_test) -> float:
        pass

class RFRegressorBuilder(ModelBuilder):
    """
    Builds a RandomForestRegressor model.
    """
    def build_model(self, df: pd.DataFrame) -> RandomForestRegressor:
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        model = RandomForestRegressor(random_state=101)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test) -> float:
        return model.score(X_test, y_test)

class XGRegressorBuilder(ModelBuilder):
    """
    Builds an XGBRegressor model.
    """
    def build_model(self, df: pd.DataFrame) -> XGBRegressor:
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        model = XGBRegressor(random_state=101)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test) -> float:
        return model.score(X_test, y_test)

# Example usage
df = DataLoaderStrategy().handle_data("../data/Final.csv")
rfr_builder = RFRegressorBuilder()
rfr_model = rfr_builder.build_model(df)
rfr_score = rfr_builder.evaluate_model(rfr_model, *DataDivideStrategy().split_data(df)[1:])

xgb_builder = XGRegressorBuilder()
xgb_model = xgb_builder.build_model(df)
xgb_score = xgb_builder.evaluate_model(xgb_model, *DataDivideStrategy().split_data(df)[1:])

print(f"RandomForestRegressor Score: {rfr_score}")
print(f"XGBoostRegressor Score: {xgb_score}")
