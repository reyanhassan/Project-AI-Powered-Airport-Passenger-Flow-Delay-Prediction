"""Unit tests for the machine learning delay prediction pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import clean_delay_data, load_delay_dataset, split_delay_data
from ml.train import save_best_model, train_all_models


class MLPipelineTest(unittest.TestCase):
    """Check data preparation and model training behavior."""

    def test_clean_delay_data_handles_missing_values(self) -> None:
        raw_data = load_delay_dataset()
        cleaned_data = clean_delay_data(raw_data)

        self.assertFalse(cleaned_data.isna().any().any())
        self.assertIn("delayed", cleaned_data.columns)
        self.assertTrue(set(cleaned_data["delayed"].unique()).issubset({0, 1}))

    def test_split_delay_data_returns_train_and_test_sets(self) -> None:
        cleaned_data = clean_delay_data(load_delay_dataset())
        X_train, X_test, y_train, y_test = split_delay_data(cleaned_data)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_train_all_models_returns_required_metrics(self) -> None:
        models, metrics_data = train_all_models()

        self.assertEqual(
            set(models.keys()),
            {"Logistic Regression", "Decision Tree", "Random Forest"},
        )
        self.assertEqual(len(metrics_data), 3)

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertTrue(metrics_data[metric].between(0, 1).all())

    def test_save_best_model_writes_joblib_file(self) -> None:
        models, metrics_data = train_all_models()

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "best_model.joblib"
            best_model_name, saved_path = save_best_model(
                models,
                metrics_data,
                model_path,
            )

            self.assertIn(best_model_name, models)
            self.assertTrue(saved_path.exists())


if __name__ == "__main__":
    unittest.main()
