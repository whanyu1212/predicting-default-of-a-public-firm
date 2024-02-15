import pickle
from typing import Any, Dict

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
from lightgbm import LGBMClassifier, LGBMModel
from mlflow.exceptions import MlflowException
from optuna.trial import Trial
from sklearn.metrics import average_precision_score, make_scorer


class HyperparameterTuner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        # create customized scorer/metric to be used in the objective function
        self.pr_auc_scorer = make_scorer(
            average_precision_score, greater_is_better=True, needs_proba=True
        )

    def create_or_get_experiment(name: str) -> str:
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
        model: LGBMModel, trial: Trial, params: Dict[str, Any], pr_auc: float
    ):
        """
        Log the model, params, and mean accuracy from mlflow
        experiments.

        Args:
            model (LGBMModel): the lightgbm trained every trial
            trial (Trial): the optuna trial
            params (Dict[str, Any]): the parameters used for the lightgbm model
            mean_accuracy (float): the mean accuracy of the model for every trial
        """
        # logs the model, params, and pr_auc of a trial
        mlflow.lightgbm.log_model(model, "lightgbm_default_prediction_model")
        mlflow.log_params(params)
        mlflow.log_metric("PR_AUC", pr_auc)
        # storing a pickled version of the best model
        trial.set_user_attr(key="best_booster", value=pickle.dumps(model))

    def objective(self, trial):
        params = {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "num_class": 7,
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

        y_pred = lgbm_cl.predict_proba(self.X_val)

        return self.pr_auc_scorer(self.y_val, y_pred)

    def create_optuna_study(self):
        pass
