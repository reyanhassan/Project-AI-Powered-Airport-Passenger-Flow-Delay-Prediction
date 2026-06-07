"""Train and compare machine learning models for flight delay prediction."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

try:
    from .preprocessing import prepare_delay_data
except ImportError:
    from preprocessing import prepare_delay_data


def build_logistic_regression_model(preprocessor) -> Pipeline:
    """Create a Logistic Regression model with preprocessing included."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_logistic_regression(X_train, y_train, preprocessor) -> Pipeline:
    """Train Logistic Regression for the delay prediction task."""

    model = build_logistic_regression_model(preprocessor)
    model.fit(X_train, y_train)
    return model


def run_logistic_regression_training() -> float:
    """Train Logistic Regression and return its test accuracy."""

    X_train, X_test, y_train, y_test, preprocessor = prepare_delay_data()
    model = train_logistic_regression(X_train, y_train, preprocessor)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Logistic Regression accuracy: {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    run_logistic_regression_training()
