# API Usage Examples

This directory contains examples and tests for using the Candidate Scoring API directly without the web interface.

## Files

- `api_examples.py`: Contains example functions for using each predictor type (dummy, ridge, and language model)
- `test_api_examples.py`: Contains tests for all API functionality using mocked responses

## Environment Setup

1. Create a `.env` file in the root directory with the following variables:
```env
API_BASE_URL=http://localhost:8000
LM_API_BASE_URL=http://localhost:5001/v1
LM_API_KEY=not-needed
```

2. Install the required dependencies:
```bash
pip install requests python-dotenv pytest responses
```

## Running Examples

To run the example usage of all predictors:

```bash
python examples/api_examples.py
```

This will:
1. Print available predictor types
2. Print available models for each predictor
3. Run example predictions using each predictor type

## Running Tests

To run the tests:

```bash
pytest examples/test_api_examples.py -v
```

The tests use the `responses` library to mock API calls, so they don't require a running API server.

## API Usage

The examples demonstrate three main functionalities:

1. Getting available predictor types:
```python
from examples.api_examples import get_available_models
models = get_available_models()
```

2. Getting available models per predictor:
```python
from examples.api_examples import get_available_models_per_predictor
models_per_predictor = get_available_models_per_predictor()
```

3. Calculating match scores:
```python
from examples.api_examples import calculate_match

# Example with dummy predictor
result = calculate_match(
    predictor_type="dummy",
    candidate_description="Python developer with 5 years of experience",
    vacancy_description="Looking for a senior Python developer"
)

# Example with language model predictor
result = calculate_match(
    predictor_type="lm",
    candidate_description="Python developer with ML experience",
    vacancy_description="Looking for an ML engineer",
    hr_comment="Strong technical skills",
    predictor_parameters={
        "api_base_url": "http://localhost:5001/v1",
        "api_key": "not-needed",
        "model": "QuantFactory/Meta-Llama-3-8B-GGUF"
    }
)
``` 