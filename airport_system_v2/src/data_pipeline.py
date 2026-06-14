"""Dataset generation, loading, cleaning, and feature engineering.

This module owns everything related to the airline-delay tabular dataset that
feeds the machine-learning models. It can synthesise a realistic dataset when no
CSV is supplied, clean raw data (missing values, duplicates, type coercion), and
expose the column groups used throughout the rest of the application.

The synthetic generator uses a logistic "delay score" so that the label is
sampled from a probability rather than a hard rule. This produces realistic mixed
cases (on-time flights in bad weather, delayed flights in clear weather) and gives
the ML models a genuine signal to learn instead of a trivial threshold.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths and schema
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "airline_delay.csv"

TARGET_COLUMN = "delayed"
ID_COLUMN = "flight_id"

CATEGORICAL_FEATURES = [
    "airline",
    "origin_airport",
    "destination_airport",
    "flight_day",
    "weather_condition",
]
NUMERIC_FEATURES = [
    "scheduled_hour",
    "passenger_count",
    "previous_delay_minutes",
    "security_wait_minutes",
    "gate_changes",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
ALL_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# Domain vocabularies — also reused by the prediction form and the live flight board.
AIRLINES = ["PIA", "AirBlue", "SereneAir", "FlyJinnah", "Qatar", "Emirates", "Turkish"]
AIRPORTS = ["ISB", "KHI", "LHE", "DXB", "DOH", "IST", "JFK", "LHR", "SIN", "BKK"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm"]

# Bias weights driving the synthetic label. Kept as module constants so the
# prediction tool can show users *why* a flight is predicted to be delayed.
_AIRLINE_BIAS = {
    "PIA": 0.10, "AirBlue": 0.03, "SereneAir": 0.06, "FlyJinnah": 0.02,
    "Qatar": -0.04, "Emirates": -0.05, "Turkish": -0.01,
}
_AIRPORT_BIAS = {
    "ISB": 0.02, "KHI": 0.07, "LHE": 0.04, "DXB": -0.03, "DOH": -0.04,
    "IST": 0.03, "JFK": 0.08, "LHR": 0.06, "SIN": -0.05, "BKK": 0.01,
}
_WEATHER_BIAS = {"Clear": -0.25, "Cloudy": -0.05, "Rain": 0.24, "Fog": 0.36, "Storm": 0.55}
_WEATHER_WEIGHTS = [0.52, 0.22, 0.15, 0.07, 0.04]
_HOUR_WEIGHTS = {
    5: 0.05, 6: 0.07, 7: 0.09, 8: 0.11, 9: 0.08, 10: 0.05, 11: 0.04, 12: 0.04,
    13: 0.04, 14: 0.04, 15: 0.05, 16: 0.07, 17: 0.10, 18: 0.10, 19: 0.08,
    20: 0.05, 21: 0.03, 22: 0.01,
}


# --------------------------------------------------------------------------- #
# Synthetic dataset generation
# --------------------------------------------------------------------------- #
def generate_synthetic_dataset(rows: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """Generate a realistic airline-delay dataset of ``rows`` flights.

    The categorical and numeric features feed a logistic delay-score model whose
    output is sampled to produce the binary ``delayed`` label. A small amount of
    label noise and missing values is injected so the downstream cleaning step is
    meaningful and the models cannot reach unrealistic 100% accuracy.

    Args:
        rows: Number of flight records to generate.
        random_seed: Seed for reproducible output.

    Returns:
        A :class:`pandas.DataFrame` with the full :data:`ALL_COLUMNS` schema.
    """
    rng = random.Random(random_seed)
    hours = list(_HOUR_WEIGHTS.keys())
    hour_weights = list(_HOUR_WEIGHTS.values())

    records = []
    for index in range(1, rows + 1):
        airline = rng.choice(AIRLINES)
        origin = rng.choice(AIRPORTS)
        destination = rng.choice([a for a in AIRPORTS if a != origin])
        flight_day = rng.choice(DAYS)
        scheduled_hour = rng.choices(hours, weights=hour_weights, k=1)[0]
        weather = rng.choices(WEATHER_CONDITIONS, weights=_WEATHER_WEIGHTS, k=1)[0]

        passenger_count = int(max(55, min(rng.gauss(165, 38), 285)))

        previous_delay = max(0.0, rng.gauss(14, 18))
        if rng.random() < 0.12:  # occasional badly-delayed inbound aircraft
            previous_delay += rng.uniform(30, 90)
        previous_delay_minutes = int(min(previous_delay, 180))

        peak = 1 if scheduled_hour in {7, 8, 17, 18, 19} else 0
        security_wait = rng.gauss(15, 7) + peak * rng.uniform(4, 12) + max(0, passenger_count - 170) / 12
        security_wait_minutes = int(min(max(4, security_wait), 90))

        gate_changes = 0
        if weather in {"Rain", "Fog", "Storm"} and rng.random() < 0.28:
            gate_changes += 1
        if previous_delay_minutes > 50 and rng.random() < 0.25:
            gate_changes += 1
        if rng.random() < 0.05:
            gate_changes += 1
        gate_changes = min(gate_changes, 4)

        weekend_pressure = 0.08 if flight_day in {"Friday", "Sunday"} else 0.0
        evening_pressure = 0.10 if scheduled_hour >= 17 else 0.0
        noise = rng.gauss(0, 0.45)

        delay_score = (
            -1.15
            + _AIRLINE_BIAS[airline]
            + _AIRPORT_BIAS[origin]
            + _AIRPORT_BIAS[destination] * 0.55
            + _WEATHER_BIAS[weather]
            + previous_delay_minutes * 0.018
            + security_wait_minutes * 0.014
            + gate_changes * 0.20
            + passenger_count * 0.0015
            + weekend_pressure
            + evening_pressure
            + noise
        )
        delay_probability = 1 / (1 + math.exp(-delay_score))
        delayed = int(rng.random() < delay_probability)
        if rng.random() < 0.08:  # intentional label noise — real data is messy
            delayed = 1 - delayed

        records.append({
            "flight_id": f"FL{index:04d}",
            "airline": airline,
            "origin_airport": origin,
            "destination_airport": destination,
            "flight_day": flight_day,
            "scheduled_hour": scheduled_hour,
            "weather_condition": weather,
            "passenger_count": passenger_count,
            "previous_delay_minutes": previous_delay_minutes,
            "security_wait_minutes": security_wait_minutes,
            "gate_changes": gate_changes,
            "delayed": delayed,
        })

    frame = pd.DataFrame(records, columns=ALL_COLUMNS)

    # Inject a little missing data so the cleaning step has work to do.
    nullable = ["passenger_count", "previous_delay_minutes", "security_wait_minutes", "weather_condition"]
    for column in nullable:
        mask = frame.sample(frac=0.018, random_state=random_seed).index
        frame.loc[mask, column] = None

    return frame


def get_or_create_dataset(path: str | Path = RAW_DATA_PATH, rows: int = 3000) -> pd.DataFrame:
    """Load the dataset from ``path``, generating and saving it if absent.

    Args:
        path: Location of the CSV file.
        rows: Number of rows to generate when the file does not yet exist.

    Returns:
        The raw (uncleaned) dataset as a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = generate_synthetic_dataset(rows=rows)
        frame.to_csv(path, index=False)
        return frame
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Cleaning and feature engineering
# --------------------------------------------------------------------------- #
def summarize_dataset(data: pd.DataFrame) -> dict[str, object]:
    """Return high-level statistics used for the before/after cleaning display.

    Args:
        data: Any version of the dataset (raw or cleaned).

    Returns:
        A dictionary with row/column counts, missing-cell and duplicate totals,
        and the overall delay rate.
    """
    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "delay_rate": float(data[TARGET_COLUMN].mean()) if TARGET_COLUMN in data else 0.0,
    }


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset: drop duplicates, impute missing values, fix types.

    Categorical gaps are filled with the column mode; numeric gaps with the
    median. The target is coerced to a 0/1 integer.

    Args:
        data: Raw dataset containing at least :data:`ALL_COLUMNS`.

    Returns:
        A cleaned DataFrame with no missing values in the modelled columns.

    Raises:
        ValueError: If any required column is missing from ``data``.
    """
    missing = [c for c in ALL_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cleaned = data[ALL_COLUMNS].copy().drop_duplicates()

    for column in CATEGORICAL_FEATURES:
        mode = cleaned[column].mode(dropna=True)
        cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        # passenger/hour/etc. are integers conceptually — keep them tidy.
        cleaned[column] = cleaned[column].round().astype(int)

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].fillna(cleaned[TARGET_COLUMN].mode().iloc[0]).astype(int)

    return cleaned.reset_index(drop=True)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataset into the feature matrix ``X`` and target ``y``.

    Args:
        data: Cleaned dataset.

    Returns:
        A ``(X, y)`` tuple where ``X`` contains only :data:`FEATURE_COLUMNS`.
    """
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].copy()
