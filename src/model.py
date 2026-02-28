import os
import mlflow
import optuna
import warnings
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature

# Отключаем логи Optuna и FutureWarning
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.ERROR)  # Только ошибки
warnings.filterwarnings("ignore", category=FutureWarning)


def predict(model, X) -> np.ndarray:
    """Получаем предсказания."""

    return model.predict(X)


def eval_metrics(y, y_pred) -> tuple[np.float64]:
    """Считаем метрики."""

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2


def gradient_boosting_cv_train(X_train, y_train, X_test, y_test) -> tuple[GradientBoostingRegressor, dict]:
    """Специализированная функция для обучения градиентного бустинга. Использует GridSearchCV для подбора гиперпараметров."""

    params = {
        "n_estimators": [1, 10, 50, 100],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "loss": ["squared_error", "huber"],
    }

    model = GradientBoostingRegressor(random_state=42)
    model = GridSearchCV(model, params, cv=3, n_jobs=1, scoring="r2")
    model.fit(X_train, y_train.to_numpy().flatten())
    model = model.best_estimator_

    return model, {
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "subsample": model.subsample,
        "loss": model.loss,
        "max_features": model.max_features,
    }


def xgb_train_optuna(X_train, y_train, X_test, y_test) -> tuple[XGBRegressor, dict]:
    """Специализированная функция для обучения XGB. Использует Optuna для подбора гиперпараметров."""

    def objective(trial):
        params = {
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'random_state': 42,
            'early_stopping_rounds': 50,
        }

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        y_pred = model.predict(X_test)
        metric = r2_score(y_test, y_pred)

        return metric

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100,  timeout=600)
    model = XGBRegressor(**study.best_params)
    model.fit(X_train, y_train)

    return model, study.best_params


def train_and_save_results(train_func, predict_func, log_model_func, X_train, y_train, X_test, y_test, experiment_name: str) -> None:
    """Обучаем модель и сохраняем всю информацию об эксперименте, используя MLflow."""

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Обучаем модель и сохраняем параметры
        model, params_dict = train_func(X_train, y_train, X_test, y_test)
        mlflow.log_params(params_dict)

        # Получаем предсказания на тесте и сохраняем метрики
        y_pred = predict(model, X_test)
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Сохраняем модель
        signature = infer_signature(X_train.to_pandas(), y_pred)
        log_model_func(model, name="model", signature=signature)

        # Сохраняем данные
        data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        for name, df in data.items():
            df.write_csv(f"{name}.csv")
            mlflow.log_artifact(f"{name}.csv")
            os.remove(f"{name}.csv")

