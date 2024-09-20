import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from steps.model_builder_evaluator import ConcreteBestModelFinder
from steps.data_loader_splitter import DataLoaderStrategy, DataDivideStrategy

df = DataLoaderStrategy().handle_data("../data/Final.csv")
best_model = ConcreteBestModelFinder().find_best_model(df)
print(f"The Best Model Is : {type(best_model).__name__}")