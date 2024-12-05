# test_clustering.py
"""
Unit tests for the phrase clustering application.
"""

import unittest
import numpy as np
import pandas as pd
from clustering import (
    load_json_data,
    extract_features,
    process_data,
    perform_clustering,
    create_visualizations,
    download_with_cli,
)
from unittest.mock import patch
import os
import json


class TestPhraseClustering(unittest.TestCase):
    """Test cases for phrase clustering functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.test_data = {
            "Category1": {"group_0": [["test phrase one"], ["another test phrase"]]}
        }

        # Create test JSON file
        with open("test_data.json", "w") as f:
            json.dump(cls.test_data, f)

    def test_load_json_data(self):
        """Test JSON data loading."""
        data = load_json_data("test_data.json")
        self.assertIsInstance(data, dict)
        with self.assertRaises(FileNotFoundError):
            load_json_data("nonexistent.json")

    def test_extract_features(self):
        """Test feature extraction."""
        phrases = [["test phrase"], ["another phrase"]]
        features = extract_features(phrases)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 2)
        self.assertIn("word_count", features.columns)
        self.assertIn("char_count", features.columns)

    def test_process_data(self):
        """Test data processing."""
        df, scaled = process_data(self.test_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(scaled, np.ndarray)
        self.assertEqual(len(df), len(scaled))

    def test_perform_clustering(self):
        """Test clustering functionality."""
        scaled_features = np.array([[1, 2], [1, 4], [8, 9], [8, 10]])
        clusters = perform_clustering(scaled_features, n_clusters=2)
        self.assertIsInstance(clusters, np.ndarray)
        self.assertEqual(len(clusters), 4)
        self.assertEqual(len(np.unique(clusters)), 2)

    def test_create_visualizations(self):
        """Test that visualizations save correctly."""
        scaled_features = np.array([[1, 2], [1, 4], [8, 9], [8, 10]])
        clusters = np.array([0, 0, 1, 1])
        df = pd.DataFrame(
            {
                "word_count": [1, 1, 8, 8],
                "char_count": [2, 4, 9, 10],
                "Cluster": clusters,
            }
        )
        create_visualizations(scaled_features, clusters, df)
        self.assertTrue(os.path.exists("scatter_clusters.png"))
        self.assertTrue(os.path.exists("pca_clusters.png"))
        os.remove("scatter_clusters.png")
        os.remove("pca_clusters.png")

    def test_download_with_cli(self):
        """Test downloading with AWS CLI."""
        with patch("subprocess.run") as mocked_run:
            mocked_run.return_value = None  # Mock successful execution
            download_with_cli("dummy-bucket", "dummy-file", "dummy-output")
            mocked_run.assert_called_once_with(
                "aws s3 cp s3://dummy-bucket/dummy-file dummy-output --no-sign-request",
                shell=True,
                check=True,
            )

    def test_error_handling(self):
        """Test error handling in main functions."""
        with self.assertRaises(Exception):
            extract_features(None)
        with self.assertRaises(Exception):
            perform_clustering(np.array([[]]))

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists("test_data.json"):
            os.remove("test_data.json")


def run_tests():
    """Run all unit tests and save results in unit_test_results folder."""
    try:
        # Ensure the results directory exists
        results_dir = "unit_test_results"
        os.makedirs(results_dir, exist_ok=True)

        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPhraseClustering)

        # Run tests and capture results
        with open(os.path.join(results_dir, "test_report.txt"), "w") as f:
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)

        # Print summary to console
        print("\nTest Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")

        return result.wasSuccessful()

    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return False


if __name__ == "__main__":
    run_tests()
