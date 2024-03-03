from typing import Tuple

import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler

from src.extract_financial_data import FinancialDataExtractor


class DataProcessor:
    def __init__(self, data: pd.DataFrame, cutoff_date: str):
        """
        Initialize the DataProcessor class with the given parameters.

        Args:
            data (pd.DataFrame): input raw data that needs
            to be processed
            cutoff_date (str): threshold date for filtering,
            do not consider the data prior to this cut off
        """
        self.data = data
        self.cutoff_date = pd.to_datetime(cutoff_date)
        # initialize the scaler and store it as an attribute
        self.scaler = MinMaxScaler()

    def filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter the data by the cutoff date
        Args:
            data (pd.DataFrame): input data

        Returns:
            pd.DataFrame: output data after filtering
        """
        data["Date"] = pd.to_datetime(data["Date"])
        return data[data["Date"] > self.cutoff_date]

    def one_hot_encode_categorical_columns(
        self, data: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """
        One hot encode the given column in the input data.

        Args:
            data (pd.DataFrame): input data
            column (str): column that needs encoding

        Returns:
            pd.DataFrame: output dataframe with encoded columns
        """
        df_dummies = pd.get_dummies(data[column], prefix=column)
        df_encoded = pd.concat([data, df_dummies], axis=1)
        df_encoded = df_encoded.drop(columns=[column])
        # some hard coded logic to remove the .0 from the encoded column names
        df_encoded.columns = [col.replace(".0", "") for col in df_encoded.columns]
        return df_encoded

    def winsorize_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize the numerical columns in the input data. At this
        point, unwanted columns are yet to be removed. That explains for
        the need to check dtypes.

        Args:
            data (pd.DataFrame): input data

        Returns:
            pd.DataFrame: output data with winsorization applied
        """
        for col in data.select_dtypes("number").columns:
            if col != "Y":
                data[col] = winsorize(
                    data[col], limits=[0.05, 0.05], inclusive=(True, True)
                )
        return data

    def min_max_scale_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply min-max scaling to the numerical columns in the input
        data. at this point, unwanted columns are yet to be removed.
        That explains for the need to check dtypes. The scaler is
        initialized in the __init__ method and stored as an attribute.

        Min max scaling shouldn't matter too much for tree based models, but
        it's a good practice to apply it to the data.

        Args:
            data (pd.DataFrame): input data

        Returns:
            pd.DataFrame: output data with min-max scaling applied
        """
        for col in data.select_dtypes("number").columns:
            if col != "Y":
                data[col] = self.scaler.fit_transform(data[[col]])
        return data

    def get_data_range_from_df(self, data: pd.DataFrame) -> Tuple[str, str]:
        """
        Based on the input data, get the date range. Use this date range
        for fetching oil pricing data later on.

        Args:
            data (pd.DataFrame): input data

        Returns:
            Tuple[str, str]: output date range in string format
        """
        start = (data["Date"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (data["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return start, end

    def fetch_auxiliary_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch auxiliary data (Daily oil closing price) from Yahoo
        Finance using the FinancialDataExtractor class.

        Args:
            ticker (str): ticker symbol
            start (str): start date
            end (str): end date

        Returns:
            pd.DataFrame: output data
        """
        extractor = FinancialDataExtractor(ticker, start, end)
        aux_data = extractor.extraction_flow()
        aux_data = aux_data.filter(items=["Date", "Close", "Return"])

        return aux_data

    def add_auxiliary_data(
        self, data: pd.DataFrame, aux_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge the original data with the auxiliary data.

        Args:
            data (pd.DataFrame): processed input data
            aux_data (pd.DataFrame): financial data

        Returns:
            pd.DataFrame: merged data
        """
        data_merged = data.merge(aux_data, on="Date", how="left")
        return data_merged

    def save_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for the input data.

        Args:
            data (pd.DataFrame): input data

        Returns:
            pd.DataFrame: summary statistics
        """
        data.describe().to_csv("./data/processed/summary_statistics.csv")

    def process_flow(self, column: str) -> pd.DataFrame:
        """
        Combine the entire data processing flow.

        Args:
            column (str): column that needs encoding

        Returns:
            pd.DataFrame: overall processed data
        """
        df = self.data.copy()
        df = self.filter_data_by_date(df)
        df = self.one_hot_encode_categorical_columns(df, column)
        df = self.winsorize_numerical_columns(df)
        self.save_summary_statistics(df)
        df = self.min_max_scale_numerical_columns(df)
        start, end = self.get_data_range_from_df(df)
        aux_data = self.fetch_auxiliary_data("BZ=F", start, end)
        df_combined = self.add_auxiliary_data(df, aux_data)

        return df_combined


# if __name__ == "__main__":
#     data = pd.read_csv("./data/raw/input.csv")
#     data = input_data_schema.validate(data)
#     data_processor = DataProcessor(data, "2000-1-1")
#     processed_data = data_processor.process_flow("INDUSTRY2")
#     processed_data.to_csv("./data/processed/processed_input.csv", index=False)
#     print(processed_data)
