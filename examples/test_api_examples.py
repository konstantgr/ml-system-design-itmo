"""Tests for the Candidate Scoring API examples."""

import json
import os
from typing import Dict

import pytest
import responses
from dotenv import load_dotenv

from examples.api_examples import (
    API_BASE_URL,
    calculate_match,
    get_available_models,
    get_available_models_per_predictor,
)

# Load environment variables
load_dotenv()


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        # Mock available models endpoint
        rsps.add(
            responses.GET,
            f"{API_BASE_URL}/available-models",
            json={"predictor_types": ["dummy", "ridge", "lm"]},
            status=200,
        )

        # Mock available models per predictor endpoint
        rsps.add(
            responses.GET,
            f"{API_BASE_URL}/available-models-per-predictor",
            json={
                "models": {
                    "dummy": ["dummy-model-v1"],
                    "ridge": ["ridge-model-v1"],
                    "lm": ["bartowski/Qwen2.5-32B-Instruct-GGUF/Qwen2.5-32B-Instruct-Q2_K_L.gguf"],
                }
            },
            status=200,
        )

        # Mock match endpoint for all predictors
        def request_callback(request):
            payload = json.loads(request.body.decode())
            predictor_type = payload["predictor_type"]
            scores = {"dummy": 0.5, "ridge": 0.75, "lm": 0.85}
            response_body = {"score": scores[predictor_type], "description": f"{predictor_type.upper()} prediction"}
            return (200, {"Content-Type": "application/json"}, json.dumps(response_body))

        rsps.add_callback(
            responses.POST,
            f"{API_BASE_URL}/match",
            callback=request_callback,
            content_type="application/json",
        )

        yield rsps


def test_get_available_models(mock_api_responses):
    """Test getting available models."""
    result = get_available_models()
    assert "predictor_types" in result
    assert set(result["predictor_types"]) == {"dummy", "ridge", "lm"}


def test_get_available_models_per_predictor(mock_api_responses):
    """Test getting available models per predictor."""
    result = get_available_models_per_predictor()
    assert "models" in result
    models = result["models"]
    assert "dummy" in models
    assert "ridge" in models
    assert "lm" in models
    assert models["lm"] == ["bartowski/Qwen2.5-32B-Instruct-GGUF/Qwen2.5-32B-Instruct-Q2_K_L.gguf"]


@pytest.mark.parametrize(
    "predictor_type,expected_score",
    [
        ("dummy", 0.5),
        ("ridge", 0.75),
        ("lm", 0.85),
    ],
)
def test_calculate_match(mock_api_responses, predictor_type: str, expected_score: float):
    """Test calculating match scores for different predictors."""
    result = calculate_match(
        predictor_type=predictor_type,
        candidate_description="Test candidate",
        vacancy_description="Test vacancy",
        hr_comment="Test comment",
        predictor_parameters=(
            {
                "api_base_url": os.getenv("LM_API_BASE_URL", "http://localhost:5001/v1"),
                "api_key": os.getenv("LM_API_KEY", "not-needed"),
                "model": "bartowski/Qwen2.5-32B-Instruct-GGUF/Qwen2.5-32B-Instruct-Q2_K_L.gguf",
            }
            if predictor_type == "lm"
            else None
        ),
    )

    assert isinstance(result, Dict)
    assert "score" in result
    assert "description" in result
    assert result["score"] == expected_score
