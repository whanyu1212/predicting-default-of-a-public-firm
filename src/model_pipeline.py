import numpy as np
import pandas as pd

from src.hyperparameter_tuning import HyperparameterTuner


class ModelPipeline:
    def __init__(
        self, df, test_splitting_date, validation_splitting_date, target_column
    ):
        self.df = df
        self.test_splitting_date = test_splitting_date
        self.validation_splitting_date = validation_splitting_date
        self.target_column = target_column

    def naive_timeseries_splitting(self):
        conditions = [
            (self.df["Date"] < self.validation_splitting_date),
            (self.df["Date"] >= self.validation_splitting_date)
            & (self.df["Date"] < self.test_splitting_date),
            (self.df["Date"] >= self.test_splitting_date),
        ]

        choices = ["train", "val", "test"]
        self.df["indicator"] = np.select(conditions, choices, default="test")
        train, val, test = (
            self.df[self.df["indicator"] == "train"],
            self.df[self.df["indicator"] == "val"],
            self.df[self.df["indicator"] == "test"],
        )
        return train, val, test

    def remove_unwanted_features(self, *dfs):
        unwanted_columns = ["Company_name", "Date", "CompNo", "indicator"]
        return [df.drop(columns=unwanted_columns) for df in dfs]

    def train_model(self, train, val):
        X_train = train.drop(columns=[self.target_column])
        y_train = train[self.target_column]

        X_val = val.drop(columns=[self.target_column])
        y_val = val[self.target_column]

        # Create an instance of HyperparameterTuner with the tuning data
        tuner = HyperparameterTuner(
            X_train,
            y_train,
            X_val,
            y_val,
            "./config/catalog.yaml",
        )

        # Run the hyperparameter tuning
        model = tuner.create_optuna_study("lightgbm_model", "1")

        # Create a new LGBMClassifier with the best parameters and fit it
        # model = LGBMClassifier(**best_params)
        # model.fit(X_train, y_train)

        return model

    # def predict(self, model, test):
    #     X_test = test.drop(columns=[self.target_column])
    #     y_test = test[self.target_column]
    #     y_pred = model.predict(X_test)
    #     return y_pred, y_test

    # def eval_model(self, y_pred, y_test):
    #     return average_precision_score(y_test, y_proba)

    def run_pipeline(self):
        train, val, test = self.naive_timeseries_splitting()
        train, val, test = self.remove_unwanted_features(train, val, test)
        model = self.train_model(train, val)
        # y_pred, y_test = self.predict(model, test)
        # score = self.eval_model(y_pred, y_test)
        return model


if __name__ == "__main__":
    df = pd.read_csv("./data/processed/processed_input.csv")
    model_pipeline = ModelPipeline(df, "2020-1-1", "2015-1-1", "Y")
    model_pipeline.run_pipeline()
