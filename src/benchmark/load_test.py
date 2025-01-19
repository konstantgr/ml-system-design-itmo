"""Load testing script for the Candidate Scoring API using Locust."""

import json
import os

from dotenv import load_dotenv
from locust import HttpUser, between, task

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class CandidateScoringUser(HttpUser):
    """User class for load testing the Candidate Scoring API."""

    wait_time = between(1, 3)  # Random wait between tasks

    def on_start(self):
        """Initialize test data."""
        self.test_data = {
            "dummy": {
                "predictor_type": "dummy",
                "candidate_description": "Python developer with 5 years of experience",
                "vacancy_description": "Looking for a senior Python developer",
                "hr_comment": "The candidate has good technical skills",
                "predictor_parameters": None,
            },
            "ridge": {
                "predictor_type": "ridge",
                "candidate_description": "Python developer with 5 years of experience in ML",
                "vacancy_description": "Looking for a senior ML engineer with Python skills",
                "hr_comment": "The candidate shows strong ML expertise",
                "predictor_parameters": None,
            },
        }

    @task(1)
    def get_available_models(self):
        """Test the available-models endpoint."""
        with self.client.get("/available-models", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get available models: {response.status_code}")

    @task(1)
    def get_available_models_per_predictor(self):
        """Test the available-models-per-predictor endpoint."""
        with self.client.get("/available-models-per-predictor", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get available models per predictor: {response.status_code}")

    @task(3)
    def calculate_match_dummy(self):
        """Test the match endpoint with dummy predictor."""
        self._calculate_match("dummy")

    @task(3)
    def calculate_match_ridge(self):
        """Test the match endpoint with ridge predictor."""
        self._calculate_match("ridge")

    def _calculate_match(self, predictor_type: str):
        """Helper method to test the match endpoint."""
        payload = self.test_data[predictor_type]
        with self.client.post("/match", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "score" in result and "description" in result:
                        response.success()
                    else:
                        response.failure("Response missing required fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Failed to calculate match: {response.status_code}")


if __name__ == "__main__":
    # This script is meant to be run using the Locust command-line interface
    pass
