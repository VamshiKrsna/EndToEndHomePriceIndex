import os
from abc import ABC, abstractmethod
from typing import Union
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor  # Ensure XGBoost is installed
import pandas as pd
from data_loader_splitter import DataLoaderStrategy, DataDivideStrategy

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

class BestModelFinder(ABC):
    """
    Abstract base class for finding the best model.
    """
    @abstractmethod
    def find_best_model(self, df: pd.DataFrame) -> Union[LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor]:
        pass

    def save_best_model(self, model, model_name: str):
        """
        Saves the best model's information to a CSV file in the "models/" directory.
        """
        model_info = {'Model Name': [model_name], 'Model Parameters': [model.get_params()]}
        df = pd.DataFrame(model_info)
        os.makedirs('models', exist_ok=True)
        df.to_csv(f'models/{model_name}_info.csv', index=False)
        print(f"Model saved to models/{model_name}_info.csv")


class ConcreteBestModelFinder(BestModelFinder):
    """
    Concrete implementation to find the best model among available models.
    """

    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegressionBuilder(),
            "DecisionTreeRegressor": DTRegressorBuilder(),
            "RandomForestRegressor": RFRegressorBuilder(),
            "XGBRegressor": XGRegressorBuilder()
        }

    def find_best_model(self, df: pd.DataFrame) -> Union[LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor]:
        """
        Finds the best model based on R^2 score and returns the best model.
        """
        best_model_name = None
        best_score = float('-inf')
        best_model = None

        for model_name, builder in self.models.items():
            model = builder.build_model(df)
            X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
            score = builder.evaluate_model(model, X_test, y_test)

            print(f"{model_name} Score: {score}")
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = model

        self.save_best_model(best_model, best_model_name)
        return best_model

class LinearRegressionBuilder(ModelBuilder):
    """
    Builds a Linear Regression model.
    """
    def build_model(self, df: pd.DataFrame) -> RandomForestRegressor:
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test) -> float:
        return model.score(X_test, y_test) # R^2 Score of the model

class DTRegressorBuilder(ModelBuilder):
    """
    Builds a Decision Tree Regressor model.
    """
    def build_model(self, df: pd.DataFrame) -> RandomForestRegressor:
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        model = DecisionTreeRegressor(random_state=101)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test) -> float:
        return model.score(X_test, y_test) # R^2 Score of the model

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
        return model.score(X_test, y_test) # R^2 Score of the model

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
        return model.score(X_test, y_test) # R^2 Score of the model

# Example usage - Everything works like a charm ! 
# df = DataLoaderStrategy().handle_data("../data/Final.csv")
# X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)

# lr_builder = LinearRegressionBuilder()
# lr_model = lr_builder.build_model(df)
# lr_score = lr_builder.evaluate_model(lr_model, X_test, y_test)

# dt_builder = DTRegressorBuilder()
# dt_model = dt_builder.build_model(df)
# dt_score = dt_builder.evaluate_model(dt_model, X_test, y_test)

# rfr_builder = RFRegressorBuilder()
# rfr_model = rfr_builder.build_model(df)
# rfr_score = rfr_builder.evaluate_model(rfr_model, X_test, y_test)

# xgb_builder = XGRegressorBuilder()
# xgb_model = xgb_builder.build_model(df)
# xgb_score = xgb_builder.evaluate_model(xgb_model, X_test, y_test)

# print(f"LinearRegression Score: {lr_score}")
# print(f"DecisionTreeRegressor Score: {dt_score}")   
# print(f"RandomForestRegressor Score: {rfr_score}")
# print(f"XGBoostRegressor Score: {xgb_score}")

# finder = ConcreteBestModelFinder()
# best_model = finder.find_best_model(df)
# print(f"The Best Model Is : {type(best_model).__name__}")