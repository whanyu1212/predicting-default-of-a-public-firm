import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

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
        self.X_train = train.drop(columns=[self.target_column])
        self.y_train = train[self.target_column]

        self.X_val = val.drop(columns=[self.target_column])
        self.y_val = val[self.target_column]

        # Create an instance of HyperparameterTuner with the tuning data
        tuner = HyperparameterTuner(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
        )

        # Run the hyperparameter tuning
        best_params = tuner.create_optuna_study("lightgbm_model", "1")

        return best_params

    def create_model_with_best_params(self, best_params):
        lgbm_cl = LGBMClassifier(**best_params)
        lgbm_cl.fit(self.X_train, self.y_train)
        return lgbm_cl

    def eval_model_performance(self, model, test):
        X_test = test.drop(columns=[self.target_column])
        y_test = test[self.target_column]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # A'PyFuncModel' loaded from optuna
        # does not have a 'predict_proba' method
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics

    def get_feature_importance(self, model):
        feature_importance = pd.Series(
            model.feature_importances_, index=self.X_val.columns
        )
        return feature_importance.sort_values(ascending=False)

    def run_pipeline(self):
        train, val, test = self.naive_timeseries_splitting()
        train, val, test = self.remove_unwanted_features(train, val, test)
        model = self.train_model(train, val)
        lgbm_cl = self.create_model_with_best_params(model)
        metrics = self.eval_model_performance(lgbm_cl, test)
        logger.info(f"Model performance: {metrics}")
        feature_importance = self.get_feature_importance(lgbm_cl)
        logger.info(f"Feature importance with scores: {feature_importance}")


# if __name__ == "__main__":
#     df = pd.read_csv("./data/processed/processed_input.csv")
#     model_pipeline = ModelPipeline(df, "2020-1-1", "2015-1-1", "Y")
#     model_pipeline.run_pipeline()
