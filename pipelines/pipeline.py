# from steps.fred_data_ingestion import ingest_fred_data
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(root))

from steps.data_loader_splitter import DataLoaderStrategy, DataDivideStrategy
from steps.model_builder_evaluator import ConcreteBestModelFinder 

df = DataLoaderStrategy().handle_data("../data/Final.csv")  
best_model = ConcreteBestModelFinder().find_best_model(df)
print(f"The Best Model Is : {type(best_model).__name__}")