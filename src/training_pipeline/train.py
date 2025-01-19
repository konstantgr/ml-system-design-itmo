import os

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using custom metrics:
    - ≤ 5% of predictions differ by more than 1 point
    - ≤ 20% of predictions differ by more than 0.5 points
    """
    # Calculate absolute differences
    abs_diff = np.abs(y_true - y_pred)

    # Calculate percentages of predictions that differ by more than 1 and 0.5 points
    diff_more_than_1 = np.mean(abs_diff > 1.0) * 100
    diff_more_than_05 = np.mean(abs_diff > 0.5) * 100

    # Check if metrics meet the requirements
    meets_1point_req = diff_more_than_1 <= 5.0
    meets_05point_req = diff_more_than_05 <= 20.0

    results = {
        "pct_diff_more_than_1": diff_more_than_1,
        "pct_diff_more_than_0.5": diff_more_than_05,
        "meets_1point_requirement": meets_1point_req,
        "meets_0.5point_requirement": meets_05point_req,
        "meets_all_requirements": meets_1point_req and meets_05point_req,
    }

    return results


def train_model():
    # Load processed data
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Make predictions on both train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate standard metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nStandard Metrics:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    # Evaluate custom metrics
    print("\nCustom Accuracy Metrics:")

    # Training set evaluation
    train_results = evaluate_predictions(y_train, y_train_pred)
    print("\nTraining Set Results:")
    print(f"Predictions differing by >1 point: {train_results['pct_diff_more_than_1']:.2f}% (target: ≤5%)")
    print(f"Predictions differing by >0.5 points: {train_results['pct_diff_more_than_0.5']:.2f}% (target: ≤20%)")
    print(f"Meets all requirements: {train_results['meets_all_requirements']}")

    # Test set evaluation
    test_results = evaluate_predictions(y_test, y_test_pred)
    print("\nTest Set Results:")
    print(f"Predictions differing by >1 point: {test_results['pct_diff_more_than_1']:.2f}% (target: ≤5%)")
    print(f"Predictions differing by >0.5 points: {test_results['pct_diff_more_than_0.5']:.2f}% (target: ≤20%)")
    print(f"Meets all requirements: {test_results['meets_all_requirements']}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/vacancy_matcher.joblib")

    # Save evaluation results
    evaluation_results = {
        "train_metrics": train_results,
        "test_metrics": test_results,
        "standard_metrics": {"train_mse": train_mse, "test_mse": test_mse, "train_r2": train_r2, "test_r2": test_r2},
    }

    joblib.dump(evaluation_results, "models/evaluation_results.joblib")

    return model, evaluation_results


if __name__ == "__main__":
    model, results = train_model()
