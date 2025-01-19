"""Examples of using the Candidate Scoring API directly."""

import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
LM_API_KEY = os.getenv("LM_API_KEY", "not-needed")
LM_API_BASE_URL = os.getenv("LM_API_BASE_URL", "http://localhost:5001/v1")


def get_available_models() -> Dict:
    """Get available predictor types."""
    response = requests.get(f"{API_BASE_URL}/available-models")
    return response.json()


def get_available_models_per_predictor() -> Dict:
    """Get available models for each predictor type."""
    response = requests.get(f"{API_BASE_URL}/available-models-per-predictor")
    return response.json()


def calculate_match(
    predictor_type: str,
    candidate_description: str,
    vacancy_description: str,
    hr_comment: str,
    predictor_parameters: Optional[Dict] = None,
) -> Dict:
    """Calculate match score between vacancy and candidate."""
    payload = {
        "predictor_type": predictor_type,
        "candidate_description": candidate_description,
        "vacancy_description": vacancy_description,
        "hr_comment": hr_comment,
        "predictor_parameters": predictor_parameters,
    }
    response = requests.post(f"{API_BASE_URL}/match", json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.json()}")
    return response.json()


def example_dummy_predictor():
    """Example of using the dummy predictor."""
    result = calculate_match(
        predictor_type="dummy",
        candidate_description="Python developer with 5 years of experience",
        vacancy_description="Looking for a senior Python developer",
        hr_comment="The candidate has good technical skills",
    )
    print("\nDummy Predictor Result:")
    print(f"Score: {result['score']}")
    print(f"Description: {result['description']}")


def example_ridge_predictor():
    """Example of using the ridge predictor."""
    result = calculate_match(
        predictor_type="ridge",
        candidate_description="Python developer with 5 years of experience in ML",
        vacancy_description="Looking for a senior ML engineer with Python skills",
        hr_comment="The candidate shows strong ML expertise",
    )
    print("\nRidge Predictor Result:")
    print(f"Score: {result['score']}")
    print(f"Description: {result['description']}")


def example_lm_predictor():
    """Example of using the language model predictor."""
    predictor_parameters = {
        "api_base_url": LM_API_BASE_URL,
        "api_key": LM_API_KEY,
        "model": "bartowski/Qwen2.5-32B-Instruct-GGUF/Qwen2.5-32B-Instruct-Q2_K_L.gguf",
    }
    result = calculate_match(
        predictor_type="lm",
        candidate_description="Python developer with 5 years of experience in ML and deep learning",
        vacancy_description="Looking for a senior ML engineer with Python and deep learning expertise",
        hr_comment="The candidate showed strong technical skills during the interview",
        predictor_parameters=predictor_parameters,
    )
    print("\nLanguage Model Predictor Result:")
    print(f"Score: {result['score']}")
    print(f"Description: {result['description']}")


if __name__ == "__main__":
    # Print available models
    print("Available predictor types:")
    print(get_available_models())

    print("\nAvailable models per predictor:")
    print(get_available_models_per_predictor())

    # Run examples
    example_dummy_predictor()
    example_ridge_predictor()
    example_lm_predictor()
