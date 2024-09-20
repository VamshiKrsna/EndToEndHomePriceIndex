import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from steps.data_loader_splitter import DataLoaderStrategy, DataDivideStrategy
from steps.model_builder_evaluator import ConcreteBestModelFinder

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(project_root)
# csv_path = os.path.join(project_root, "data", "Final.csv")

# df = DataLoaderStrategy().handle_data(csv_path)
# best_model = ConcreteBestModelFinder().find_best_model(df)
# print(f"The Best Model Is : {type(best_model).__name__}")