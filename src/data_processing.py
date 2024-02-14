import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from src.util.data_schema import input_data_schema


class DataProcessor:
    def __init__(self, data, cutoff_date):
        self.data = data
        self.cutoff_date = cutoff_date

    def filter_data_by_date(self, data):
        return data[data["Date"] < self.cutoff_date]

    def one_hot_encode_categorical_columns(self, data):
        industry_encoder = OneHotEncoder()
        industry_encoded = industry_encoder.fit_transform(data[["INDUSTRY2"]])
        industry_encoded_dense = industry_encoded.toarray()
        industry_df = pd.DataFrame(
            industry_encoded_dense,
            columns=industry_encoder.get_feature_names_out(["INDUSTRY2"]),
        )
        # seems to have trailing .0 in the column names
        industry_df.columns = industry_df.columns.str.replace(".0", "")

        data = pd.concat([data, industry_df], axis=1)

        data = data.drop("INDUSTRY2", axis=1)

        return data

    def winsorize_numerical_columns(self, data):
        for col in data.select_dtypes("number").columns:
            data[col] = winsorize(data[col], limits=[0.05, 0.05], inclusive=(True, True))
        return data

    def min_max_scale_numerical_columns(self, data):
        scaler = MinMaxScaler()
        for col in data.select_dtypes("number").columns:
            data[col] = scaler.fit_transform(data[[col]])
        return data

    def label_encode_response_variable(self, data):
        label_encoder = LabelEncoder()
        data["Y"] = label_encoder.fit_transform(data["Y"])
        return data

    def remove_unwanted_columns(self, data):
        data.drop(["Company_name", "Date", "CompNo"], axis=1, inplace=True)
        return data

    def process_flow(self):
        df = self.data.copy()
        df = self.filter_data_by_date(df)
        df = self.one_hot_encode_categorical_columns(df)
        df = self.winsorize_numerical_columns(df)
        df = self.min_max_scale_numerical_columns(df)
        df = self.label_encode_response_variable(df)
        df = self.remove_unwanted_columns(df)
        return df


if __name__ == "__main__":
    data = pd.read_csv("./data/raw/input.csv")
    data = input_data_schema.validate(data)
    data_processor = DataProcessor(data, "2000-1-1")
    processed_data = data_processor.process_flow()
    print(processed_data.describe())
