import random
import string

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mstats
from sklearn.preprocessing import MinMaxScaler

from src.data_processing import DataProcessor


@pytest.fixture
def dp():
    data = pd.DataFrame(
        {
            "Date": pd.date_range(start="1/1/2020", end="1/10/2020"),
            "A": np.random.randint(0, 100, 10),
            "B": [
                "".join(random.choices(string.ascii_lowercase, k=5)) for _ in range(10)
            ],
            "Y": np.random.randint(0, 100, 10),
        }
    )
    return DataProcessor(data, "2020-01-05")


def test_filter_data_by_date(dp):
    filtered_data = dp.filter_data_by_date(dp.data)
    assert all(filtered_data["Date"] > pd.to_datetime("2020-01-05"))


def test_one_hot_encode_categorical_columns(dp):
    original_data = dp.data.copy()
    encoded_data = dp.one_hot_encode_categorical_columns(dp.data, "B")
    assert (
        encoded_data.shape[1]
        == original_data.shape[1] + len(original_data["B"].unique()) - 1
    )
    for category in original_data["B"].unique():
        assert f"B_{category}" in encoded_data.columns


def test_winsorize_numerical_columns(dp):
    winsorized_data = dp.winsorize_numerical_columns(dp.data)
    assert (
        mstats.winsorize(dp.data["A"], limits=[0.05, 0.05]).min()
        == winsorized_data["A"].min()
    )
    assert (
        mstats.winsorize(dp.data["A"], limits=[0.05, 0.05]).max()
        == winsorized_data["A"].max()
    )


def test_min_max_scale_numerical_columns(dp):
    scaled_data = dp.min_max_scale_numerical_columns(dp.data)
    scaler = MinMaxScaler()
    assert np.allclose(
        scaler.fit_transform(dp.data[["A"]]), scaled_data["A"].values.reshape(-1, 1)
    )


def test_get_data_range_from_df(dp):
    start, end = dp.get_data_range_from_df(dp.data)
    assert start == (dp.data["Date"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert end == (dp.data["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
