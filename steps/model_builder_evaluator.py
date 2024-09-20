import os
import sys
import pickle
import datetime
from abc import ABC, abstractmethod
from typing import Union
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor  # Ensure XGBoost is installed
import pandas as pd
from data_loader_splitter import DataLoaderStrategy, DataDivideStrategy

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(root))  

class ModelBuilder(ABC):
    """
    Abstract base class for building models.
    """
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def build_model(self, df: pd.DataFrame):
        X_train, X_test, y_train, y_test = DataDivideStrategy().split_data(df)
        model = self.model_cls()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test) -> float:
        return model.score(X_test, y_test)  # R^2 Score of the model

class BestModelFinder(ABC):
    """
    Abstract base class for finding the best model.
    """
    @abstractmethod
    def find_best_model(self, df: pd.DataFrame) -> Union[LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor]:
        pass

    def save_best_model_info(self, model, model_name: str):
        """
        Saves the best model's information to a CSV file in the "models/" directory.
        """
        model_info = {'Model Name': [model_name], 'Model Parameters': [model.get_params()]}
        dt_info = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df = pd.DataFrame(model_info)
        models_path = os.path.join(os.getcwd(),"..",'saved_models')
        os.makedirs(models_path, exist_ok=True)
        df.to_csv(f'{models_path}/{model_name}_{dt_info}_info.csv', index=False)
        print(f"Model saved to models/{model_name}_{dt_info}_info.csv")
    
    def save_best_model(self, model, model_name: str):
        """
        Saves the best model to a pickle file in the "models/" directory.
        """
        model_path = os.path.join(os.getcwd(),"..",'saved_models')
        os.makedirs(model_path, exist_ok=True)
        with open(f'{model_path}/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to models/{model_name}.pkl")    


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

        self.save_best_model_info(best_model, best_model_name)
        self.save_best_model(best_model, best_model_name)
        return best_model

class LinearRegressionBuilder(ModelBuilder):
    """
    Builds a Linear Regression model.
    """
    def __init__(self):
        super().__init__(LinearRegression)

class DTRegressorBuilder(ModelBuilder):
    """
    Builds a Decision Tree Regressor model.
    """
    def __init__(self):
        super().__init__(DecisionTreeRegressor)

class RFRegressorBuilder(ModelBuilder):
    """
    Builds a RandomForestRegressor model.
    """
    def __init__(self):
        super().__init__(RandomForestRegressor)

class XGRegressorBuilder(ModelBuilder):
    """
    Builds an XGBRegressor model.
    """
    def __init__(self):
        super().__init__(XGBRegressor)


# Example usage
df = DataLoaderStrategy().handle_data("../data/Final.csv")
finder = ConcreteBestModelFinder()
best_model = finder.find_best_model(df)
print(f"The Best Model Is : {type(best_model).__name__}")