"""Machine-learning pipeline for flight-delay prediction.

Trains and compares three classifiers (Logistic Regression, Decision Tree,
Random Forest), each wrapped in an sklearn :class:`~sklearn.pipeline.Pipeline`
that bundles preprocessing (one-hot encoding + scaling) with the estimator. The
best model by F1 score is persisted with joblib so the simulation and prediction
tabs can reuse it without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]


def build_preprocessor() -> ColumnTransformer:
    """Create the column transformer used by every model pipeline.

    One-hot encodes categorical features and standard-scales numeric ones.

    Returns:
        A configured (unfitted) :class:`~sklearn.compose.ColumnTransformer`.
    """
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """Build the three model pipelines keyed by display name.

    Returns:
        Mapping of model name to an unfitted preprocessing+estimator pipeline.
    """
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)),
        ]),
    }


def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the four headline classification metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, object]:
    """Train all models, evaluate them, and select the best by F1 score.

    Args:
        X: Feature matrix (raw columns — preprocessing happens inside each pipeline).
        y: Binary target series.
        test_size: Fraction of data held out for evaluation.
        progress_callback: Optional ``fn(fraction, label)`` invoked during training
            so the UI can drive a progress bar.

    Returns:
        A results dictionary with keys: ``models`` (fitted pipelines),
        ``metrics`` (DataFrame), ``confusion_matrices`` (name -> 2x2 array),
        ``best_model_name``, ``best_model``, ``feature_importance`` (DataFrame for
        the best model) and the split test data (``X_test``, ``y_test``).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = build_models()
    fitted: dict[str, Pipeline] = {}
    metrics_rows = []
    confusion_matrices: dict[str, np.ndarray] = {}

    total = len(models)
    for i, (name, pipeline) in enumerate(models.items()):
        if progress_callback:
            progress_callback(i / total, f"Training {name}…")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        fitted[name] = pipeline
        metrics_rows.append({"model": name, **_evaluate(y_test, predictions)})
        confusion_matrices[name] = confusion_matrix(y_test, predictions)

    if progress_callback:
        progress_callback(1.0, "Training complete")

    metrics = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_model_name]

    return {
        "models": fitted,
        "metrics": metrics,
        "confusion_matrices": confusion_matrices,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_importance": compute_feature_importance(best_model),
        "rf_importance": compute_feature_importance(fitted["Random Forest"]),
        "X_test": X_test,
        "y_test": y_test,
    }


def _readable_feature_names(pipeline: Pipeline) -> list[str]:
    """Map encoded feature names back to friendly labels (e.g. ``weather=Fog``)."""
    raw = pipeline.named_steps["preprocessor"].get_feature_names_out()
    pretty = []
    for name in raw:
        # ColumnTransformer prefixes with the transformer name, e.g. "categorical__weather_condition_Fog".
        stripped = name.split("__", 1)[-1]
        for col in CATEGORICAL_FEATURES:
            if stripped.startswith(col + "_"):
                stripped = f"{col} = {stripped[len(col) + 1:]}"
                break
        pretty.append(stripped)
    return pretty


def compute_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract per-feature importance/weight from a fitted pipeline.

    Uses ``feature_importances_`` for tree models and absolute coefficients for
    Logistic Regression. The signed direction is preserved so the prediction tool
    can show which features push *toward* a delay.

    Args:
        pipeline: A fitted model pipeline.

    Returns:
        DataFrame with columns ``feature``, ``importance`` (signed) and
        ``magnitude`` (absolute), sorted by magnitude descending.
    """
    names = _readable_feature_names(pipeline)
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:  # Logistic Regression coefficients
        importance = classifier.coef_[0]

    frame = pd.DataFrame({
        "feature": names,
        "importance": importance,
        "magnitude": np.abs(importance),
    })
    return frame.sort_values("magnitude", ascending=False).reset_index(drop=True)


def save_model(pipeline: Pipeline, path: str | Path = BEST_MODEL_PATH) -> Path:
    """Persist a fitted pipeline to disk with joblib.

    Args:
        pipeline: Fitted model pipeline to save.
        path: Destination ``.joblib`` path.

    Returns:
        The path the model was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path = BEST_MODEL_PATH) -> Pipeline | None:
    """Load a saved model pipeline, or ``None`` if it does not exist.

    Args:
        path: Source ``.joblib`` path.

    Returns:
        The loaded pipeline, or ``None`` when no model has been trained yet.
    """
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def model_exists(path: str | Path = BEST_MODEL_PATH) -> bool:
    """Return ``True`` if a trained model file is present on disk."""
    return Path(path).exists()


def predict_delay(pipeline: Pipeline, features: dict | pd.DataFrame) -> tuple[int, float]:
    """Predict delay status and probability for one or more flights.

    Args:
        pipeline: A fitted model pipeline.
        features: A single feature dict or a DataFrame of feature rows.

    Returns:
        For a single record, a ``(label, delay_probability)`` tuple. The label is
        0 (On Time) or 1 (Delayed) and the probability is for the positive class.
    """
    frame = pd.DataFrame([features]) if isinstance(features, dict) else features
    frame = frame[FEATURE_COLUMNS]
    probability = float(pipeline.predict_proba(frame)[0][1])
    label = int(probability >= 0.5)
    return label, probability


def predict_delay_batch(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return delay probabilities for a batch of flights.

    Args:
        pipeline: A fitted model pipeline.
        frame: DataFrame containing the feature columns.

    Returns:
        1-D array of positive-class (delay) probabilities, one per row.
    """
    return pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
