import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextPreprocessor:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_bert_embedding(self, text):
        # Handle NaN values
        if pd.isna(text):
            text = ""

        # Truncate text to avoid memory issues
        max_length = 512
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings[0]


def preprocess_text(candidate_description: str, vacancy_description: str, hr_comment: str) -> np.ndarray:
    """
    Preprocess input texts using BERT embeddings.

    Args:
        candidate_description (str): Description of the candidate's experience and skills
        vacancy_description (str): Description of the job vacancy requirements
        hr_comment (str): HR comments about candidate's experience (not used in current model)

    Returns:
        np.ndarray: Feature vector combining embeddings of vacancy and candidate descriptions
    """
    preprocessor = TextPreprocessor()

    # Get embeddings for each text
    vacancy_emb = preprocessor.get_bert_embedding(vacancy_description)
    candidate_emb = preprocessor.get_bert_embedding(candidate_description)
    # Note: HR comment is not used in the current model version

    # Combine embeddings (matching training data structure)
    features = np.concatenate([vacancy_emb, candidate_emb])
    return features


def prepare_dataset(data_path):
    """Prepare dataset for training."""
    df = pd.read_csv(data_path)

    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Get embeddings for job descriptions
    print("Processing job descriptions...")
    job_embeddings = []
    for desc in tqdm(df["job_description"]):
        emb = preprocessor.get_bert_embedding(desc)
        job_embeddings.append(emb)

    # Get embeddings for resume texts
    print("Processing resume texts...")
    resume_embeddings = []
    for text in tqdm(df["resume_text"]):
        emb = preprocessor.get_bert_embedding(text)
        resume_embeddings.append(emb)

    # Convert to numpy arrays
    X_job = np.array(job_embeddings)
    X_resume = np.array(resume_embeddings)

    # Create feature matrix by concatenating job and resume embeddings
    X = np.concatenate([X_job, X_resume], axis=1)

    # Create target variable from project_rating
    def convert_rating(rating):
        if pd.isna(rating):
            return np.nan
        try:
            num, den = rating.split("/")
            return float(num) / float(den)
        except:
            return float(rating) if not pd.isna(rating) else np.nan

    y = df["project_rating"].apply(convert_rating)

    # Remove rows with NaN targets
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    return X, y


if __name__ == "__main__":
    X, y = prepare_dataset("data/synthetic_dataset.csv")
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)
