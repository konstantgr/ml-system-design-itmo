# Baseline Model Evaluation

## Model Architecture

The baseline model consists of two main components:

1. **Text Embedding Model**
   - Model: BERT (bert-base-uncased)
   - Input: Job descriptions and candidate resume texts
   - Output: 768-dimensional embeddings for each text
   - Implementation: Using Hugging Face's transformers library

2. **Regression Model**
   - Model: Ridge Regression
   - Input: Concatenated BERT embeddings (1536 dimensions)
   - Output: Predicted match score (0-5 scale)
   - Hyperparameters: alpha=1.0

### Model Architecture Diagram

```mermaid
graph TD
    subgraph Input
        JD[Job Description]
        RT[Resume Text]
    end

    subgraph BERT_Processing
        B1[BERT Encoder]
        B2[BERT Encoder]
        JD --> B1
        RT --> B2
        B1 --> E1[Embedding 768d]
        B2 --> E2[Embedding 768d]
    end

    subgraph Feature_Engineering
        E1 --> C[Concatenate]
        E2 --> C
        C --> F[Final Features 1536d]
    end

    subgraph Regression
        F --> R[Ridge Regression]
        R --> S[Match Score 0-5]
    end

    style JD fill:#f9f,stroke:#333
    style RT fill:#f9f,stroke:#333
    style B1 fill:#bbf,stroke:#333
    style B2 fill:#bbf,stroke:#333
    style E1 fill:#dfd,stroke:#333
    style E2 fill:#dfd,stroke:#333
    style F fill:#dfd,stroke:#333
    style R fill:#fdb,stroke:#333
    style S fill:#f96,stroke:#333
```

## Data Processing
- Maximum sequence length: 512 tokens
- Text preprocessing: BERT tokenization
- Feature engineering: Concatenation of job description and resume text embeddings
- Target variable: Project ratings converted to float (e.g., "4/5" → 0.8)

## Model Performance

### Standard Metrics
| Metric | Training Set | Test Set |
|--------|-------------|-----------|
| MSE    | 0.0274      | 2.2054    |
| R2     | 0.9792      | -0.1170   |

### Custom Accuracy Metrics

#### Training Set Results
- Predictions differing by >1 point: 0.00% (target: ≤5%) ✅
- Predictions differing by >0.5 points: 0.00% (target: ≤20%) ✅
- Meets all requirements: ✅

#### Test Set Results
- Predictions differing by >1 point: 60.00% (target: ≤5%) ❌
- Predictions differing by >0.5 points: 100.00% (target: ≤20%) ❌
- Meets all requirements: ❌

## Analysis

The baseline model using resume text shows similar overfitting issues as the previous version:
1. Strong performance on training data (MSE: 0.0274, R2: 0.9792) but poor generalization to test data
2. Worse test set performance (MSE: 2.2054, R2: -0.1170)
3. The negative R2 score on the test set indicates the model performs worse than a horizontal line
4. The model still fails to meet the custom accuracy requirements on the test set
