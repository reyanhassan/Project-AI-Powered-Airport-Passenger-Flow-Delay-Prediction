"""Train and compare machine learning models for flight delay prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from .preprocessing import PROJECT_ROOT, prepare_delay_data
except ImportError:
    from preprocessing import PROJECT_ROOT, prepare_delay_data


MODEL_DIR = PROJECT_ROOT / "ml" / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_delay_model.joblib"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "model_metrics.csv"


def build_logistic_regression_model(preprocessor) -> Pipeline:
    """Create a Logistic Regression model with preprocessing included."""

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_logistic_regression(X_train, y_train, preprocessor) -> Pipeline:
    """Train Logistic Regression for the delay prediction task."""

    model = build_logistic_regression_model(preprocessor)
    model.fit(X_train, y_train)
    return model


def build_decision_tree_model(preprocessor) -> Pipeline:
    """Create a Decision Tree model with preprocessing included."""

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                DecisionTreeClassifier(max_depth=5, random_state=42),
            ),
        ]
    )


def train_decision_tree(X_train, y_train, preprocessor) -> Pipeline:
    """Train a Decision Tree for the delay prediction task."""

    model = build_decision_tree_model(preprocessor)
    model.fit(X_train, y_train)
    return model


def build_random_forest_model(preprocessor) -> Pipeline:
    """Create a Random Forest model with preprocessing included."""

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                ),
            ),
        ]
    )


def train_random_forest(X_train, y_train, preprocessor) -> Pipeline:
    """Train a Random Forest for the delay prediction task."""

    model = build_random_forest_model(preprocessor)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model_name: str, model: Pipeline, X_test, y_test) -> dict[str, float | str]:
    """Calculate the required classification metrics for one model."""

    predictions = model.predict(X_test)

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 3),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 3),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 3),
    }


def train_all_models() -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Train all required models and return them with their metrics."""

    X_train, X_test, y_train, y_test, preprocessor = prepare_delay_data()
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train, preprocessor),
        "Decision Tree": train_decision_tree(X_train, y_train, preprocessor),
        "Random Forest": train_random_forest(X_train, y_train, preprocessor),
    }

    metrics = [
        evaluate_model(model_name, model, X_test, y_test)
        for model_name, model in models.items()
    ]

    return models, pd.DataFrame(metrics)


def save_best_model(
    models: dict[str, Pipeline],
    metrics_data: pd.DataFrame,
    model_path: str | Path = BEST_MODEL_PATH,
) -> tuple[str, Path]:
    """Save the model with the highest F1 score using joblib."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_row = metrics_data.sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    ).iloc[0]
    best_model_name = str(best_row["model"])

    joblib.dump(models[best_model_name], model_path)
    return best_model_name, model_path


def save_model_metrics(
    metrics_data: pd.DataFrame,
    output_path: str | Path = METRICS_PATH,
) -> Path:
    """Save model comparison metrics as a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data.to_csv(output_path, index=False)
    return output_path


def run_training_pipeline() -> tuple[pd.DataFrame, str]:
    """Train, evaluate, save metrics, and save the best delay model."""

    models, metrics_data = train_all_models()
    metrics_path = save_model_metrics(metrics_data)
    best_model_name, model_path = save_best_model(models, metrics_data)

    print("\nModel comparison:")
    print(metrics_data.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Best model: {best_model_name}")
    print(f"Saved best model to: {model_path}")

    return metrics_data, best_model_name


def run_logistic_regression_training() -> float:
    """Train Logistic Regression and return its test accuracy."""

    X_train, X_test, y_train, y_test, preprocessor = prepare_delay_data()
    model = train_logistic_regression(X_train, y_train, preprocessor)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Logistic Regression accuracy: {accuracy:.3f}")
    return accuracy


def run_decision_tree_training() -> float:
    """Train Decision Tree and return its test accuracy."""

    X_train, X_test, y_train, y_test, preprocessor = prepare_delay_data()
    model = train_decision_tree(X_train, y_train, preprocessor)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Decision Tree accuracy: {accuracy:.3f}")
    return accuracy


def run_random_forest_training() -> float:
    """Train Random Forest and return its test accuracy."""

    X_train, X_test, y_train, y_test, preprocessor = prepare_delay_data()
    model = train_random_forest(X_train, y_train, preprocessor)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Random Forest accuracy: {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    run_training_pipeline()
