# Customer Support Ticket Classification

## Description
This project aims to automatically classify customer support tickets into predefined categories such as Billing Inquiry, Cancellation Request, Product Inquiry, Refund Request, and Technical Issue. The goal is to improve customer service efficiency by routing tickets to the appropriate department using machine learning and NLP techniques.

## Dataset
- **Source:** /content/customer_support_tickets.csv
- **Format:** CSV 
- **Columns:**
  - `ticket_id` – Unique identifier for each support ticket
  - `text` – The content of the customer ticket
  - `category` – The class label for the ticket (Billing, Cancellation, etc.)
- **Number of Samples:** 1694
- **Class Distribution:**
  | Class                  | Support |
  |------------------------|---------|
  | Billing inquiry        | 327     |
  | Cancellation request    | 339     |
  | Product inquiry         | 328     |
  | Refund request          | 351     |
  | Technical issue         | 349     |

## Preprocessing
- Remove punctuation and special characters
- Convert text to lowercase
- Tokenization
- Stopword removal (optional)
- Vectorization (TF-IDF / Word embeddings / BERT embeddings)

## Modeling
- **Algorithms Used:** Logistic Regression, Random Forest, XGBoost, BERT (for NLP tasks)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Performance (baseline):**
  - Accuracy: 20%
  - Class-wise F1-scores: 0.17 – 0.23

## Next Steps / Improvements
- Handle class imbalance using oversampling or weighted loss
- Use advanced NLP models (BERT, RoBERTa) for better text understanding
- Feature engineering and hyperparameter tuning
- Perform error analysis via confusion matrix to reduce misclassifications

## Usage
```python
import pandas as pd

# Load dataset
data = pd.read_csv("customer_tickets.csv")

# Example: Get tickets for a specific category
billing_tickets = data[data['category'] == 'Billing inquiry']
print(billing_tickets.head())

# Model Classification Report

##  Project Overview
This notebook contains the evaluation results of a classification model. The goal of this analysis is to measure the model's ability to correctly identify positive cases while minimizing false alarms and missed detections.

##  Performance Metrics
Based on the confusion matrix analysis, the model achieved the following performance:

| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Accuracy** | **97.2%** | Overall correct predictions |
| **Precision** | **93.3%** | Reliability of positive predictions |
| **Recall** | **86.5%** | Ability to find all actual positive cases |
| **F1-Score** | **89.8%** | Balance between Precision and Recall |

## 🧩 Confusion Matrix Breakdown
The model was tested on **1,830 total samples**.

- **True Positives (TP):** 224
- **True Negatives (TN):** 1555
- **False Positives (FP):** 16 (Type I Error)
- **False Negatives (FN):** 35 (Type II Error)



##  Observations 
1. **High Accuracy:** The model is performing exceptionally well overall.
2. **Recall Improvement:** The model is currently missing 35 positive cases. If this is a high-stakes environment (like medical or fraud detection), the next step is to tune the decision threshold to improve Recall.
3. **Data Balance:** There is a significant class imbalance (1,571 negatives vs. 259 positives). Future versions may require oversampling or SMOTE techniques.

## Improving Recall

The initial model showed a low recall (averaging ~20%). To improve performance, I implemented:
- **Hyperparameter Tuning:** Used `RandomizedSearchCV` to find the optimal settings for a Random Forest Classifier.
- **Class Balancing:** Utilized `class_weight='balanced'` to ensure the model gives equal importance to all inquiry types.
- **Metric-Driven Search:** Optimized for the **Macro F1-Score** to ensure a balanced performance across all 5 classes.
