from typing import Tuple

import joblib
import numpy as np

from src.platform.base_predictor import BasePredictor
from src.training_pipeline.data_preprocessing import preprocess_text


class RidgePredictor(BasePredictor):
    """A predictor that uses the trained Ridge model for candidate-vacancy matching."""

    def __init__(self):
        """Initialize the predictor by loading the trained model."""
        self.model = joblib.load("models/vacancy_matcher.joblib")

    def predict(
        self,
        candidate_description: str,
        vacancy_description: str,
        hr_comment: str,
    ) -> Tuple[float, str]:
        """
        Predict match score using the trained Ridge model.

        Args:
            candidate_description (str): Description of the candidate's experience and skills
            vacancy_description (str): Description of the job vacancy requirements
            hr_comment (str): HR comments about candidate's experience

        Returns:
            Tuple[float, str]: Score between 0 and 5 and a description of the match
        """
        # Preprocess the input texts
        features = preprocess_text(candidate_description, vacancy_description, hr_comment)
        features = features.reshape(1, -1)  # Reshape for single prediction

        # Make prediction
        score = self.model.predict(features)[0]

        # Ensure score is between 0 and 5
        score = np.clip(score * 5, 0, 5)  # Scale up prediction to 0-5 range
        score = round(score, 2)

        # Generate description based on score
        if score >= 4:
            description = "Excellent match! The candidate's profile strongly aligns with the position requirements."
        elif score >= 3:
            description = "Good match. The candidate has many of the required qualifications."
        elif score >= 2:
            description = "Moderate match. Some qualifications align with the requirements."
        else:
            description = "Limited match. The candidate's profile shows minimal alignment with the requirements."

        return score, description

    def get_available_models(self) -> Tuple[str]:
        """Return the available model version."""
        return ("ridge-model-v1",)
