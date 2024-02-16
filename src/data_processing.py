import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler

from src.extract_financial_data import FinancialDataExtractor
from src.util.data_schema import input_data_schema


class DataProcessor:
    def __init__(self, data, cutoff_date):
        self.data = data
        self.cutoff_date = pd.to_datetime(cutoff_date)
        self.scaler = MinMaxScaler()

    def filter_data_by_date(self, data):
        data["Date"] = pd.to_datetime(data["Date"])
        return data[data["Date"] > self.cutoff_date]

    def one_hot_encode_categorical_columns(self, data, column):
        df_dummies = pd.get_dummies(data[column], prefix=column)
        df_encoded = pd.concat([data, df_dummies], axis=1)
        df_encoded = df_encoded.drop(columns=[column])
        df_encoded.columns = [col.replace(".0", "") for col in df_encoded.columns]
        return df_encoded

    def winsorize_numerical_columns(self, data):
        data_copy = data.copy()
        for col in data_copy.select_dtypes("number").columns:
            if col != "Y":
                data_copy[col] = winsorize(
                    data_copy[col], limits=[0.05, 0.05], inclusive=(True, True)
                )
        return data_copy

    def min_max_scale_numerical_columns(self, data):
        data_copy = data.copy()
        for col in data_copy.select_dtypes("number").columns:
            if col != "Y":
                data_copy[col] = self.scaler.fit_transform(data_copy[[col]])
        return data_copy

    def get_data_range_from_df(self, data):
        start = (data["Date"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (data["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return start, end

    def fetch_auxiliary_data(self, ticker, start, end):
        extractor = FinancialDataExtractor(ticker, start, end)
        aux_data = extractor.extraction_flow()
        aux_data = aux_data.filter(items=["Date", "Close", "Return"])

        return aux_data

    def add_auxiliary_data(self, data, aux_data):
        data_merged = data.merge(aux_data, on="Date", how="left")
        return data_merged

    def process_flow(self, column):
        df = self.data.copy()
        df = self.filter_data_by_date(df)
        df = self.one_hot_encode_categorical_columns(df, column)
        df = self.winsorize_numerical_columns(df)
        df = self.min_max_scale_numerical_columns(df)
        start, end = self.get_data_range_from_df(df)
        aux_data = self.fetch_auxiliary_data("BZ=F", start, end)
        df_combined = self.add_auxiliary_data(df, aux_data)
        return df_combined


if __name__ == "__main__":
    data = pd.read_csv("./data/raw/input.csv")
    data = input_data_schema.validate(data)
    data_processor = DataProcessor(data, "2000-1-1")
    processed_data = data_processor.process_flow("INDUSTRY2")
    processed_data.to_csv("./data/processed/processed_input.csv", index=False)
    print(processed_data)
