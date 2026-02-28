import os
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.sdk import DAG
from src.model import *
from src.preprocessing import *
from pathlib import Path


# Константы
DAG_DIR = Path(__file__).parent
DATA_DIR = DAG_DIR / "../data"
TEMP_DATA_DIR = DAG_DIR / "../temp_data"
TEMP_DATA_DIR.mkdir(exist_ok=True)
DATASET_PATH = DATA_DIR / "possum.csv"
VERBOSE = True
DROP_NULLS = True
DROP_DUPLICATED = True
COLS_CONVERSION = {
    "cols_to_delete": ["case"],
    "cols_to_float": ["age", "foot_length"],
    "cols_to_str": ["site"],
}
# Конфигурация Polars
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_hide_dataframe_shape(True)


# Функции задач
def load_task(**context) -> str:
    df = load(DATASET_PATH)

    output_path = TEMP_DATA_DIR / "data.parquet"
    df.write_parquet(output_path)

    return str(output_path)


def preprocess_task(**context) -> str:
    ti = context["ti"]
    input_path = ti.xcom_pull(task_ids="load_data")

    df = pl.read_parquet(input_path)
    df = preprocess(df, COLS_CONVERSION, DROP_NULLS, DROP_DUPLICATED)

    df.write_parquet(input_path)

    return input_path


def split_task(**context) -> dict[str, str]:
    ti = context["ti"]
    input_path = ti.xcom_pull(task_ids="preprocess_data")

    df = pl.read_parquet(input_path)
    os.remove(input_path)

    X_train, X_test, y_train, y_test = split(df, y="age")

    X_train.write_parquet(TEMP_DATA_DIR / "X_train.parquet")
    X_test.write_parquet(TEMP_DATA_DIR / "X_test.parquet")
    y_train.write_parquet(TEMP_DATA_DIR / "y_train.parquet")
    y_test.write_parquet(TEMP_DATA_DIR / "y_test.parquet")

    return {
        "X_train": str(TEMP_DATA_DIR / "X_train.parquet"),
        "X_test": str(TEMP_DATA_DIR / "X_test.parquet"),
        "y_train": str(TEMP_DATA_DIR / "y_train.parquet"),
        "y_test": str(TEMP_DATA_DIR / "y_test.parquet"),
    }


def scale_task(**context) -> dict[str, str]:
    ti = context["ti"]
    paths = ti.xcom_pull(task_ids="split_data")

    X_train = pl.read_parquet(paths["X_train"])
    X_test = pl.read_parquet(paths["X_test"])

    X_train, X_test = scale(
        X_train,
        X_test,
        num_features=X_train.columns[:9],
        cat_features=X_train.columns[9:],
        scaler=StandardScaler,
    )

    X_train.write_parquet(paths["X_train"])
    X_test.write_parquet(paths["X_test"])

    return paths


def memory_optimize_task(**context) -> dict[str, str]:
    ti = context["ti"]
    paths = ti.xcom_pull(task_ids="scale_data")

    X_train = pl.read_parquet(paths["X_train"])
    X_test = pl.read_parquet(paths["X_test"])

    X_train, X_test = memory_optimize(X_train, X_test)

    X_train.write_parquet(paths["X_train"])
    X_test.write_parquet(paths["X_test"])

    return paths


def analysis_task(**context) -> dict[str, str]:
    ti = context["ti"]
    paths = ti.xcom_pull(task_ids="memory_optimize")

    X_train = pl.read_parquet(paths["X_train"])
    X_test = pl.read_parquet(paths["X_test"])

    analysis(X_train, verbose=VERBOSE)
    analysis(X_test, verbose=VERBOSE)

    return paths


def train_task(**context) -> None:
    ti = context["ti"]
    paths = ti.xcom_pull(task_ids="analysis")

    train_and_save_results(
        gradient_boosting_cv_train,
        predict,
        mlflow.sklearn.log_model,
        pl.read_parquet(paths["X_train"]),
        pl.read_parquet(paths["y_train"]),
        pl.read_parquet(paths["X_test"]),
        pl.read_parquet(paths["y_test"]),
        "possum",
    )
    for path in paths.values():
        os.remove(path)


with DAG(
    "possum",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    description="Processing + training pipeline for possum age prediction",
    schedule=timedelta(days=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["processing_and_learning"],
) as dag:
    load_data_op = PythonOperator(
        task_id="load_data",
        python_callable=load_task,
    )

    preprocess_data_op = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task,
    )

    split_data_op = PythonOperator(
        task_id="split_data",
        python_callable=split_task,
    )

    scale_data_op = PythonOperator(
        task_id="scale_data",
        python_callable=scale_task,
    )

    memory_optimize_op = PythonOperator(
        task_id="memory_optimize",
        python_callable=memory_optimize_task,
    )

    analysis_op = PythonOperator(
        task_id="analysis",
        python_callable=analysis_task,
    )

    train_model_op = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
    )

    # Зависимости
    load_data_op >> preprocess_data_op >> split_data_op >> scale_data_op >> memory_optimize_op >> analysis_op >> train_model_op