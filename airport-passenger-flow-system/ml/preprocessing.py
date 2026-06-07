"""Data preprocessing utilities for flight delay prediction.

This module keeps the machine learning preparation steps simple and reusable:
load the dataset, clean missing values, encode categorical columns, and split
the data into training and testing sets.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "airline_delay_sample.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "airline_delay_cleaned.csv"

TARGET_COLUMN = "delayed"
UNUSED_COLUMNS = ["flight_id"]
CATEGORICAL_COLUMNS = [
    "airline",
    "origin_airport",
    "destination_airport",
    "flight_day",
    "weather_condition",
]
NUMERIC_COLUMNS = [
    "scheduled_hour",
    "passenger_count",
    "previous_delay_minutes",
    "security_wait_minutes",
    "gate_changes",
]
REQUIRED_COLUMNS = UNUSED_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]


def create_sample_delay_dataset(output_path: str | Path = RAW_DATA_PATH) -> Path:
    """Create a small airline delay dataset when no raw dataset is available."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    airlines = ["PIA", "AirBlue", "SereneAir", "FlyJinnah"]
    airports = ["ISB", "KHI", "LHE", "DXB"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weather_values = ["Clear", "Cloudy", "Rain", "Fog"]
    hours = [5, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    rows = []
    for index in range(1, 61):
        weather = weather_values[index % len(weather_values)]
        previous_delay = (index * 7) % 58
        security_wait = 10 + ((index * 4) % 36)
        gate_changes = 2 if weather == "Fog" else int(weather == "Rain")

        delayed = int(
            weather in {"Rain", "Fog"}
            or previous_delay >= 30
            or security_wait >= 34
            or gate_changes >= 2
        )

        rows.append(
            {
                "flight_id": f"FL{index:03d}",
                "airline": airlines[index % len(airlines)],
                "origin_airport": airports[index % len(airports)],
                "destination_airport": airports[(index + 1) % len(airports)],
                "flight_day": days[index % len(days)],
                "scheduled_hour": hours[index % len(hours)],
                "weather_condition": weather,
                "passenger_count": 115 + ((index * 9) % 95),
                "previous_delay_minutes": previous_delay,
                "security_wait_minutes": security_wait,
                "gate_changes": gate_changes,
                "delayed": delayed,
            }
        )

    # Add a few missing values so students can see preprocessing in action.
    rows[10]["passenger_count"] = None
    rows[24]["previous_delay_minutes"] = None
    rows[37]["security_wait_minutes"] = None
    rows[48]["weather_condition"] = None

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def load_delay_dataset(data_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the airline delay dataset, creating a sample file if needed."""

    data_path = Path(data_path)
    if not data_path.exists():
        create_sample_delay_dataset(data_path)

    return pd.read_csv(data_path)


def _validate_columns(data: pd.DataFrame) -> None:
    """Check that all expected columns are present before training."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")


def clean_delay_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and keep only the columns used by the ML model."""

    _validate_columns(data)

    cleaned_data = data[REQUIRED_COLUMNS].copy()
    cleaned_data = cleaned_data.drop_duplicates()

    for column in CATEGORICAL_COLUMNS:
        most_common_value = cleaned_data[column].mode(dropna=True)
        fill_value = most_common_value.iloc[0] if not most_common_value.empty else "Unknown"
        cleaned_data[column] = cleaned_data[column].fillna(fill_value)

    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())

    cleaned_data[TARGET_COLUMN] = pd.to_numeric(
        cleaned_data[TARGET_COLUMN],
        errors="coerce",
    )
    cleaned_data[TARGET_COLUMN] = cleaned_data[TARGET_COLUMN].fillna(
        cleaned_data[TARGET_COLUMN].mode().iloc[0]
    )
    cleaned_data[TARGET_COLUMN] = cleaned_data[TARGET_COLUMN].astype(int)

    return cleaned_data


def save_cleaned_dataset(
    data: pd.DataFrame,
    output_path: str | Path = PROCESSED_DATA_PATH,
) -> Path:
    """Save the cleaned dataset so notebooks and dashboards can reuse it."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def build_feature_preprocessor() -> ColumnTransformer:
    """Create the encoder/scaler used before model training.

    OneHotEncoder turns text columns such as airline and weather into numeric
    columns. StandardScaler keeps numeric columns on a similar scale, which is
    especially useful for Logistic Regression.
    """

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def split_delay_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cleaned data into training and testing sets."""

    features = data.drop(columns=UNUSED_COLUMNS + [TARGET_COLUMN])
    target = data[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def prepare_delay_data(
    data_path: str | Path = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Load, clean, save, split, and prepare the transformer for ML training."""

    raw_data = load_delay_dataset(data_path)
    cleaned_data = clean_delay_data(raw_data)
    save_cleaned_dataset(cleaned_data)

    X_train, X_test, y_train, y_test = split_delay_data(
        cleaned_data,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = build_feature_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor
