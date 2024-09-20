from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate_model(self, df: pd.DataFrame) -> float:
        pass

class EvalRFR(ModelEvaluator):
    def evaluate_model(self, model) -> float: