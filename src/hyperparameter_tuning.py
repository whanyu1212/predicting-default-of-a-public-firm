import json
import pickle
import time
from typing import Any, Dict

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMModel
from loguru import logger
from mlflow.exceptions import MlflowException
from optuna.trial import Trial
from sklearn.metrics import average_precision_score

from src.util.general_utility_functions import parse_cfg


class HyperparameterTuner:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cfg_path: str,
    ):
        """
        Initialize the HyperparameterTuner class with the given
        parameters.

        Args:
            X_train (pd.DataFrame): Features of the training data
            y_train (pd.Series): Response of the training data
            X_val (pd.DataFrame): Features of the validation data
            y_val (pd.Series): Response of the validation data
        """

        # Some checks to ensure the input data is in the right format
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train should be a pandas DataFrame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train should be a pandas Series")
        if not isinstance(X_val, pd.DataFrame):
            raise ValueError("X_val should be a pandas DataFrame")
        if not isinstance(y_val, pd.Series):
            raise ValueError("y_val should be a pandas Series")

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # create customized scorer/metric to be used in the objective function
        self.cfg = parse_cfg(cfg_path)

    def create_or_get_experiment(self, name: str) -> str:
        """
        Create or get an mlflow experiment based on the experiment name
        specified.

        Args:
            name (str): name to be given to the experiment
            or name of the experiment to be retrieved

        Raises:
            ValueError: if the experiment name is not found

        Returns:
            str: experiment ID in string format
        """
        try:
            experiment_id = mlflow.create_experiment(name)
        except MlflowException:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                raise ValueError("Experiment not found.")
        return experiment_id

    def log_model_and_params(
        self, model: LGBMModel, trial: Trial, params: Dict[str, Any], pr_auc: float
    ):
        """
        Log the model, params, and mean accuracy from mlflow
        experiments.

        Args:
            model (LGBMModel): the lightgbm trained every trial
            trial (Trial): the optuna trial
            params (Dict[str, Any]): the parameters used for the lightgbm model
            pr_auc (float): the PR AUC of each trial
        """
        # logs the model, params, and pr_auc of a trial
        mlflow.lightgbm.log_model(model, f"lightgbm_model_trial_{trial.number}")
        mlflow.log_params(params)
        mlflow.log_metric("PR_AUC", pr_auc)
        # storing a pickled version of the best model
        trial.set_user_attr(key="best_booster", value=pickle.dumps(model))

    def objective(self, trial: Trial) -> float:

        experiment_id = self.create_or_get_experiment("lightgbm-optuna")

        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }

            lgbm_cl = LGBMClassifier(**params)

            lgbm_cl.fit(self.X_train, self.y_train)
            y_proba = lgbm_cl.predict_proba(self.X_val)[:, 1]

            # Calculate PR-AUC on the validation set
            pr_auc_score = average_precision_score(self.y_val, y_proba)

            self.log_model_and_params(lgbm_cl, trial, params, pr_auc_score)

        return pr_auc_score

    def create_optuna_study(
        self,
        model_name: str,
        model_version: str,
        n_trials: int = 10,
        max_retries: int = 3,
        delay: int = 5,
    ):
        study = optuna.create_study(study_name="test", direction="maximize")

        for _ in range(max_retries):
            try:
                study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)
                best_trial = study.best_trial
                best_params = best_trial.params
                break
            except Exception as e:
                print(f"An error occurred: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        else:
            raise RuntimeError("Failed to optimize the study after maximum retries")
        with open("./output/best_param.json", "w") as outfile:
            json.dump(best_params, outfile)

        experiment_id = self.create_or_get_experiment("lightgbm-optuna")
        runs_df = mlflow.search_runs(
            experiment_ids=experiment_id,
            order_by=["metrics.pr_auc DESC"],
        )
        best_run = runs_df.iloc[0]
        best_run_id = best_run["run_id"]

        try:
            _ = mlflow.register_model(
                "runs:/" + best_run_id + "/lightgbm_model", model_name
            )
        except MlflowException as e:
            logger.error(f"Failed to register model: {e}")

        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        logger.info(
            "Model loaded. the model information is as follows: {}".format(model)
        )

        return model
