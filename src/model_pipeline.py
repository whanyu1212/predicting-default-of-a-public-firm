import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.hyperparameter_tuning import HyperparameterTuner


class ModelPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        test_splitting_date: str,
        validation_splitting_date: str,
        target_column: str = "Y",
    ):
        """
        Initialize the ModelPipeline class with the given parameters.

        Args:
            df (pd.DataFrame): input data for model training
            test_splitting_date (str): cut off date for splitting the
            train/val and test data
            validation_splitting_date (str): cut off date for splitting
            the train and validation data
            target_column (str, optional): Name of the response. Defaults to "Y".
        """
        self.df = df
        self.test_splitting_date = test_splitting_date
        self.validation_splitting_date = validation_splitting_date
        self.target_column = target_column

    def naive_timeseries_splitting(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, val, test according to the cut off
        dates respectively.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            output dataframes for train, val, test
        """
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

    def remove_unwanted_features(self, *dfs: Tuple[pd.DataFrame]) -> Tuple[pd.DataFrame]:
        """
        Remove unwanted columns from the input dataframes.

        Returns:
            Tuple[pd.DataFrame]: output dataframes with
            unwanted columns removed
        """
        unwanted_columns = ["Company_name", "Date", "CompNo", "indicator"]
        return [df.drop(columns=unwanted_columns) for df in dfs]

    def train_model(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        """
        Train and tune the model to get the best hyperparameters.

        Args:
            train (pd.DataFrame): training data to fix model
            val (pd.DataFrame): validation data to tune model

        Returns:
            dict: combination of best hyperparameters
        """
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

    def create_model_with_best_params(self, best_params: dict) -> LGBMClassifier:
        """
        Initialize a model with the best hyperparameters found through
        optuna tuning.

        Args:
            best_params (dict): a dictionary of best hyperparameters (combination)

        Returns:
            LGBMClassifier: model
        """
        lgbm_cl = LGBMClassifier(**best_params)
        lgbm_cl.fit(self.X_train, self.y_train)
        with open("./models/lgbm_model.pkl", "wb") as f:
            pickle.dump(lgbm_cl, f)
        return lgbm_cl

    def eval_model_performance(
        self, model: LGBMClassifier, val: pd.DataFrame, test: pd.DataFrame
    ) -> dict:
        """
        Evaluate the model performance using a series of metrics.

        Args:
            model (LGBMClassifier): model fitted in the previous step
            test (pd.DataFrame): test set that the model has not seen before

        Returns:
            dict: a dictionary of metrics and scores
        """
        val_metrics = self.evaluate_set(model, val)
        test_metrics = self.evaluate_set(model, test)

        return val_metrics, test_metrics

    def evaluate_set(self, model: LGBMClassifier, data: pd.DataFrame) -> dict:
        """
        Evaluate the model performance on a given set using a series of
        metrics.

        Args:
            model (LGBMClassifier): model fitted in the previous step
            data (pd.DataFrame): data set to evaluate the model on

        Returns:
            dict: a dictionary of metrics and scores
        """
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, average="weighted"),
            "pr_auc": average_precision_score(y, y_pred_proba),
            "roc_auc": roc_auc_score(y, y_pred_proba),
        }

        return metrics

    def generate_confusion_matrix(
        self, model: LGBMClassifier, test: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate the confusion matrix for the test set.

        Args:
            model (LGBMClassifier): model fitted in the previous step
            test (pd.DataFrame): test set that the model has not seen before

        Returns:
            np.ndarray: confusion matrix
        """
        X_test = test.drop(columns=[self.target_column])
        y_test = test[self.target_column]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues")
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Truth", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16)
        plt.savefig("./output/confusion_matrix.png", bbox_inches="tight")

        return cm

    def get_feature_importance(self, model: LGBMClassifier) -> pd.Series:
        """
        Get the feature importance ranking from the model based on the
        validation set.

        Args:
            model (LGBMClassifier): model fitted

        Returns:
            pd.Series: series of feature importance scores
        """
        feature_importance = pd.Series(
            model.feature_importances_, index=self.X_val.columns
        )
        feature_importance_sorted = feature_importance.sort_values()

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        feature_importance_sorted.plot(kind="barh", color="skyblue")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")

        plt.savefig("./output/feature_importance.png", bbox_inches="tight")

        return feature_importance.sort_values(ascending=False)

    def run_pipeline(self) -> None:
        """Run the entire pipeline from data splitting to model
        evaluation."""
        train, val, test = self.naive_timeseries_splitting()
        train, val, test = self.remove_unwanted_features(train, val, test)
        model = self.train_model(train, val)
        lgbm_cl = self.create_model_with_best_params(model)
        val_metrics, test_metrics = self.eval_model_performance(lgbm_cl, val, test)
        logger.info(f"Model performance on val set: {val_metrics}")
        logger.info(f"Model performance on test set: {test_metrics}")
        feature_importance = self.get_feature_importance(lgbm_cl)
        logger.info(f"Feature importance with scores: {feature_importance}")
        cm = self.generate_confusion_matrix(lgbm_cl, test)
        logger.info(f"Confusion matrix: {cm}")


# if __name__ == "__main__":
#     df = pd.read_csv("./data/processed/processed_input.csv")
#     model_pipeline = ModelPipeline(df, "2020-1-1", "2015-1-1", "Y")
#     model_pipeline.run_pipeline()
