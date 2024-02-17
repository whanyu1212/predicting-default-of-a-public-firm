import pandas as pd

from src.data_processing import DataProcessor
from src.model_pipeline import ModelPipeline
from src.util.data_schema import input_data_schema

# Global variables


def data_processing_flow(data, cutoff_date, categorical_column):
    data = input_data_schema.validate(data)
    data_processor = DataProcessor(data, cutoff_date)
    processed_data = data_processor.process_flow(categorical_column)
    return processed_data


def modelling_flow(
    processed_data, test_splitting_date, validation_splitting_date, target_column
):
    model_pipeline = ModelPipeline(
        processed_data, test_splitting_date, validation_splitting_date, target_column
    )
    model_pipeline.run_pipeline()


def main():
    data = pd.read_csv("./data/raw/input.csv")
    cutoff_date = "2000-1-1"
    categorical_column = "INDUSTRY2"
    test_splitting_date = "2020-1-1"
    val_splitting_date = "2015-1-1"
    target_column = "Y"
    processed_data = data_processing_flow(data, cutoff_date, categorical_column)
    modelling_flow(
        processed_data, test_splitting_date, val_splitting_date, target_column
    )


if __name__ == "__main__":
    main()
