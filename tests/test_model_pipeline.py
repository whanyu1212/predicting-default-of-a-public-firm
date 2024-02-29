import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.model_pipeline import ModelPipeline


@pytest.fixture
def mp():
    data = pd.DataFrame(
        {
            "Date": pd.date_range(start="1/1/2020", end="1/10/2020"),
            "A": np.random.randint(0, 100, 10),
            "Y": np.random.randint(0, 100, 10),
            "Company_name": ["Company1"] * 5 + ["Company2"] * 5,
            "CompNo": np.random.randint(0, 100, 10),
            "indicator": ["train"] * 3 + ["val"] * 3 + ["test"] * 4,
        }
    )
    return ModelPipeline(data, "2020-01-07", "2020-01-04", "Y")


def test_naive_timeseries_splitting(mp):
    train, val, test = mp.naive_timeseries_splitting()
    assert len(train) == 3
    assert len(val) == 3
    assert len(test) == 4


def test_remove_unwanted_features(mp):
    train, val, test = mp.naive_timeseries_splitting()
    train, val, test = mp.remove_unwanted_features(train, val, test)
    for df in [train, val, test]:
        assert "Company_name" not in df.columns
        assert "Date" not in df.columns
        assert "CompNo" not in df.columns
        assert "indicator" not in df.columns


def test_train_model(mp):
    train, val, test = mp.naive_timeseries_splitting()
    train, val, test = mp.remove_unwanted_features(train, val, test)

    mp.X_train = train.drop(columns=[mp.target_column])
    mp.y_train = train[mp.target_column]
    mp.X_val = val.drop(columns=[mp.target_column])
    mp.y_val = val[mp.target_column]

    # Mock the HyperparameterTuner class
    with mock.patch("src.model_pipeline.HyperparameterTuner") as MockHyperparameterTuner:
        # Create a mock instance of the class
        mock_tuner = MockHyperparameterTuner.return_value

        # Set the return value of the create_optuna_study method
        mock_tuner.create_optuna_study.return_value = {
            "param1": "value1",
            "param2": "value2",
        }

        # Call the method under test
        best_params = mp.train_model(train, val)

    # Assert that the HyperparameterTuner was instantiated with the correct arguments
    MockHyperparameterTuner.assert_called_once_with(
        mp.X_train,
        mp.y_train,
        mp.X_val,
        mp.y_val,
    )

    # Assert that the create_optuna_study method was called with the correct arguments
    mock_tuner.create_optuna_study.assert_called_once_with("lightgbm_model", "1")

    # Assert that the method under test returned the correct result
    assert best_params == {"param1": "value1", "param2": "value2"}


def test_create_model_with_best_params(mp):
    train, val, test = mp.naive_timeseries_splitting()
    train, val, test = mp.remove_unwanted_features(train, val, test)

    # Set the X_train, y_train, X_val, and y_val attributes
    mp.X_train = train.drop(columns=[mp.target_column])
    mp.y_train = train[mp.target_column]
    mp.X_val = val.drop(columns=[mp.target_column])
    mp.y_val = val[mp.target_column]

    # Mock the train_model method
    with mock.patch.object(
        mp, "train_model", return_value={"param1": "value1", "param2": "value2"}
    ) as mock_train_model:
        model = mp.create_model_with_best_params(mp.train_model(train, val))
        mock_train_model.assert_called_once()

    assert isinstance(model, LGBMClassifier)


def test_get_feature_importance(mp):
    train, val, test = mp.naive_timeseries_splitting()
    train, val, test = mp.remove_unwanted_features(train, val, test)

    # Set the X_train, y_train, X_val, and y_val attributes
    mp.X_train = train.drop(columns=[mp.target_column])
    mp.y_train = train[mp.target_column]
    mp.X_val = val.drop(columns=[mp.target_column])
    mp.y_val = val[mp.target_column]

    # Mock the train_model method
    with mock.patch.object(
        mp, "train_model", return_value={"param1": "value1", "param2": "value2"}
    ) as mock_train_model:
        model = mp.create_model_with_best_params(mp.train_model(train, val))
        mock_train_model.assert_called_once()

    feature_importance = mp.get_feature_importance(model)
    assert set(feature_importance.index) == set(train.columns) - set(["Y"])
