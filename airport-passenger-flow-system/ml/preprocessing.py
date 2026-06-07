"""Data preprocessing utilities for flight delay prediction.

This module keeps the machine learning preparation steps simple and reusable:
load the dataset, clean missing values, encode categorical columns, and split
the data into training and testing sets.
"""

import math
import random
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


def _weighted_choice(rng: random.Random, options: list[tuple[object, float]]) -> object:
    """Choose one value from weighted options using beginner-friendly code."""

    total_weight = sum(weight for _, weight in options)
    random_value = rng.uniform(0, total_weight)
    running_weight = 0.0

    for value, weight in options:
        running_weight += weight
        if random_value <= running_weight:
            return value

    return options[-1][0]


def create_sample_delay_dataset(
    output_path: str | Path = RAW_DATA_PATH,
    rows: int = 3000,
    random_seed: int = 42,
) -> Path:
    """Create a larger realistic airline delay dataset if no raw data exists."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)
    airlines = ["PIA", "AirBlue", "SereneAir", "FlyJinnah", "Qatar", "Emirates", "Turkish"]
    airports = ["ISB", "KHI", "LHE", "DXB", "DOH", "IST", "JFK", "LHR", "SIN", "BKK"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weather_options = [
        ("Clear", 0.52),
        ("Cloudy", 0.22),
        ("Rain", 0.15),
        ("Fog", 0.07),
        ("Storm", 0.04),
    ]
    airline_bias = {
        "PIA": 0.10,
        "AirBlue": 0.03,
        "SereneAir": 0.06,
        "FlyJinnah": 0.02,
        "Qatar": -0.04,
        "Emirates": -0.05,
        "Turkish": -0.01,
    }
    airport_bias = {
        "ISB": 0.02,
        "KHI": 0.07,
        "LHE": 0.04,
        "DXB": -0.03,
        "DOH": -0.04,
        "IST": 0.03,
        "JFK": 0.08,
        "LHR": 0.06,
        "SIN": -0.05,
        "BKK": 0.01,
    }
    weather_bias = {
        "Clear": -0.25,
        "Cloudy": -0.05,
        "Rain": 0.24,
        "Fog": 0.36,
        "Storm": 0.55,
    }

    generated_rows = []
    for index in range(1, rows + 1):
        airline = rng.choice(airlines)
        origin_airport = rng.choice(airports)
        possible_destinations = [
            airport for airport in airports if airport != origin_airport
        ]
        destination_airport = rng.choice(possible_destinations)
        flight_day = rng.choice(days)

        # Morning and evening peaks are common in airport schedules.
        scheduled_hour = int(
            _weighted_choice(
                rng,
                [
                    (5, 0.05),
                    (6, 0.07),
                    (7, 0.09),
                    (8, 0.11),
                    (9, 0.08),
                    (10, 0.05),
                    (11, 0.04),
                    (12, 0.04),
                    (13, 0.04),
                    (14, 0.04),
                    (15, 0.05),
                    (16, 0.07),
                    (17, 0.10),
                    (18, 0.10),
                    (19, 0.08),
                    (20, 0.05),
                    (21, 0.03),
                    (22, 0.01),
                ],
            )
        )
        weather_condition = str(_weighted_choice(rng, weather_options))

        passenger_count = int(rng.gauss(165, 38))
        passenger_count = max(55, min(passenger_count, 285))

        base_previous_delay = max(0, rng.gauss(14, 18))
        if rng.random() < 0.12:
            base_previous_delay += rng.uniform(30, 90)
        previous_delay_minutes = int(min(base_previous_delay, 180))

        peak_hour_pressure = 1 if scheduled_hour in {7, 8, 17, 18, 19} else 0
        security_wait_minutes = int(
            max(
                4,
                rng.gauss(15, 7)
                + peak_hour_pressure * rng.uniform(4, 12)
                + max(0, passenger_count - 170) / 12,
            )
        )
        security_wait_minutes = min(security_wait_minutes, 90)

        gate_changes = 0
        if weather_condition in {"Rain", "Fog", "Storm"} and rng.random() < 0.28:
            gate_changes += 1
        if previous_delay_minutes > 50 and rng.random() < 0.25:
            gate_changes += 1
        if rng.random() < 0.05:
            gate_changes += 1
        gate_changes = min(gate_changes, 4)

        weekend_pressure = 0.08 if flight_day in {"Friday", "Sunday"} else 0.0
        evening_pressure = 0.10 if scheduled_hour >= 17 else 0.0
        hidden_operational_noise = rng.gauss(0, 0.45)

        # The label is sampled from probability instead of hard rules. This
        # creates realistic mixed cases, such as on-time rainy flights and
        # delayed clear-weather flights.
        delay_score = (
            -1.15
            + airline_bias[airline]
            + airport_bias[origin_airport]
            + airport_bias[destination_airport] * 0.55
            + weather_bias[weather_condition]
            + previous_delay_minutes * 0.018
            + security_wait_minutes * 0.014
            + gate_changes * 0.20
            + passenger_count * 0.0015
            + weekend_pressure
            + evening_pressure
            + hidden_operational_noise
        )
        delay_probability = 1 / (1 + math.exp(-delay_score))
        delayed = int(rng.random() < delay_probability)

        # Label noise is intentional. Real data has messy exceptions.
        if rng.random() < 0.08:
            delayed = 1 - delayed

        generated_rows.append(
            {
                "flight_id": f"FL{index:03d}",
                "airline": airline,
                "origin_airport": origin_airport,
                "destination_airport": destination_airport,
                "flight_day": flight_day,
                "scheduled_hour": scheduled_hour,
                "weather_condition": weather_condition,
                "passenger_count": passenger_count,
                "previous_delay_minutes": previous_delay_minutes,
                "security_wait_minutes": security_wait_minutes,
                "gate_changes": gate_changes,
                "delayed": delayed,
            }
        )

    # Add small amounts of missing data so preprocessing remains meaningful.
    missing_columns = [
        "passenger_count",
        "previous_delay_minutes",
        "security_wait_minutes",
        "weather_condition",
    ]
    for row in generated_rows:
        for column in missing_columns:
            if rng.random() < 0.018:
                row[column] = None

    temporary_path = output_path.with_suffix(".tmp.csv")
    pd.DataFrame(generated_rows).to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
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
