import polars as pl
import polars.selectors as cs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

"""Все функции для разведочного анализа и предобработки данных."""


def load(path: str) -> pl.DataFrame:
    """Загрузка данных с мгновенной обработкой пропусков."""

    # Загрузка данных с корретными null значениями
    null_names = ["NA", "Nan", "NAN", "None", "NULL", "null", "Null", "NaN"]
    df = pl.read_csv(path, null_values=null_names)
    df = df.with_columns(cs.float().fill_nan(None))

    return df


def cols_preprocess(df: pl.DataFrame, cols_conversion: dict[str, [str]]) -> pl.DataFrame:
    """Конверсия типов и удаление лишних столбцов за один проход."""

    # Удаление лишних столбцов
    df = df.drop(cols_conversion.get("cols_to_delete", []))

    # Конверсия типов
    dtypes_map = {}
    for col in cols_conversion.get("cols_to_int", []):
        dtypes_map[col] = pl.Int64
    for col in cols_conversion.get("cols_to_float", []):
        dtypes_map[col] = pl.Float64
    for col in cols_conversion.get("cols_to_str", []):
        dtypes_map[col] = pl.String

    if dtypes_map:
        df = df.cast(dtypes_map)

    return df


def rows_preprocess(df: pl.DataFrame, drop_nulls: bool = True, drop_duplicated: bool = True) -> pl.DataFrame:
    """Удаление строк с null и дубликатов."""

    # Удаление строк с пустымии значениями
    if drop_nulls:
        df = df.drop_nulls()
    else:
        # Пакетное заполнение пропусков по типам данных
        df = df.with_columns(
            cs.float().fill_null(0.0),
            cs.string().fill_null("null")
        )

    if drop_duplicated:
        df = df.unique(maintain_order=True)

    return df


def cat_preprocess(df: pl.DataFrame) -> pl.DataFrame:
    cat_cols = df.select(cs.string()).columns

    if not cat_cols:
        return df

    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(df.select(cat_cols))
    new_col_names = encoder.get_feature_names_out(cat_cols)
    df_encoded = pl.DataFrame(encoded_array, schema=list(new_col_names))

    return pl.concat([df.drop(cat_cols), df_encoded], how="horizontal")


def preprocess(df: pl.DataFrame, cols_conversion: dict[str, list[str]], drop_nulls: bool = True, drop_duplicated: bool = True) -> pl.DataFrame:
    df = (
        df
        .pipe(cols_preprocess, cols_conversion)
        .pipe(rows_preprocess, drop_nulls, drop_duplicated)
        .pipe(cat_preprocess)
    )
    return df


def split(df: pl.DataFrame, y: str) -> tuple[pl.DataFrame]:
    X = df.select(pl.exclude(y))
    y = df.select(pl.col(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def scale(X_train: pl.DataFrame, X_test: pl.DataFrame, num_features: list[str], cat_features: list[str], scaler) -> tuple[pl.DataFrame]:
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', scaler(), num_features),
        ('cat', 'passthrough', cat_features),
    ], verbose_feature_names_out=False
    ).set_output(transform="polars")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def memory_optimize(X_train: pl.DataFrame, X_test: pl.DataFrame) -> tuple[pl.DataFrame]:
    X_train = pl.DataFrame([s.shrink_dtype() for s in X_train])
    X_test = pl.DataFrame([s.shrink_dtype() for s in X_test])
    return X_train, X_test


def analysis(df: pl.DataFrame, verbose: bool = False) -> tuple[int, bool]:
    """Проверка наличия пропусков и дубликатов. Вывод статистических показателей."""

    null_count = df.null_count().sum_horizontal().item()
    is_duplicated_found = df.is_duplicated().any()

    if verbose:
        df_describe = df.describe()

        print(df.head(10))
        print(df_describe)
        print(f"Наличие пропусков: {null_count}")
        print(f"Наличие дубликатов: {is_duplicated_found}")

    return null_count, is_duplicated_found


if __name__ == "__main__":
    # Константы
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

    # Загрузка датасета
    path = ".\\possum.csv"
    df = load(path)

    # Препроцессинг
    df = preprocess(df, COLS_CONVERSION, DROP_NULLS, DROP_DUPLICATED)
    X_train, X_test, y_train, y_test = split(df, y="age")
    X_train, X_test = scale(X_train, X_test, num_features=X_train.columns[:9], cat_features=X_train.columns[9:], scaler=StandardScaler)
    X_train, X_test = memory_optimize(X_train, X_test)
    null_count, is_duplicated_found = analysis(X_train, verbose=VERBOSE)
    null_count, is_duplicated_found = analysis(X_test, verbose=VERBOSE)