"""Train and compare machine learning models for flight delay prediction."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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


def build_decision_tree_model(preprocessor) -> Pipeline:
    """Create a Decision Tree model with preprocessing included."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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
            ("preprocessor", preprocessor),
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
    run_logistic_regression_training()
    run_decision_tree_training()
    run_random_forest_training()
