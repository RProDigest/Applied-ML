# main.py
"""
Main script to run the K-Means clustering application.
"""

from clustering import main as clustering_main
from test_clustering import run_tests
import logging

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Run unit tests
        logger.info("Running unit tests...")
        tests_passed = run_tests()

        if tests_passed:
            logger.info("All tests passed. Running main application...")
            # Run main clustering application
            clustering_main()
        else:
            logger.error(
                "Tests failed. Please fix issues before running main application."
            )

    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
