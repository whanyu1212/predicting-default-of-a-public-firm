import pandas as pd

from src.data_processing import DataProcessor
from src.model_pipeline import ModelPipeline
from src.util.data_schema import input_data_schema
from src.util.general_utility_functions import parse_cfg


def data_processing_flow(
    data: pd.DataFrame, cutoff_date: str, categorical_column: str
) -> pd.DataFrame:
    """
    Calls the data processing class to process the data.

    Args:
        data (pd.DataFrame): raw input data
        cutoff_date (str): cut off date assigned to filter
        out irrelevant data
        categorical_column (str): column to be encoded

    Returns:
        pd.DataFrame: processed data
    """

    # Validate the input data schema and make sure they comply
    data = input_data_schema.validate(data)
    data_processor = DataProcessor(data, cutoff_date)
    processed_data = data_processor.process_flow(categorical_column)
    return processed_data


def modelling_flow(
    processed_data: pd.DataFrame,
    test_splitting_date: str,
    validation_splitting_date: str,
    target_column: str,
):
    """
    Calls the model pipeline class (which calls the hyperparameter
    tuning class as well) to train and evaluate the model.

    Args:
        processed_data (pd.DataFrame): data ready for splitting
        and modelling
        test_splitting_date (str): cut off date for splitting the
        train/val and test data
        validation_splitting_date (str): cut off date for splitting
        the train and validation data
        target_column (str): response column
    """
    model_pipeline = ModelPipeline(
        processed_data, test_splitting_date, validation_splitting_date, target_column
    )
    model_pipeline.run_pipeline()


def main() -> None:
    """Main function to chain everything together."""
    config = parse_cfg("./config/catalog.yaml")
    data = pd.read_csv(config["data"]["file_path"])
    cutoff_date = config["dates"]["cutoff_date"]
    categorical_column = config["columns"]["categorical_column"]
    test_splitting_date = config["dates"]["test_splitting_date"]
    val_splitting_date = config["dates"]["val_splitting_date"]
    target_column = config["columns"]["target_column"]
    processed_data = data_processing_flow(data, cutoff_date, categorical_column)
    modelling_flow(
        processed_data, test_splitting_date, val_splitting_date, target_column
    )


if __name__ == "__main__":
    main()
