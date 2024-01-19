# Credit Card Fraud Detection

## Overview

This project focuses on building and evaluating machine learning models for credit card fraud detection. The dataset used contains transactions labeled as fraudulent or non-fraudulent.

## Files

- **ML.py:** Python script containing the machine learning code.
- **creditcard.csv:** Dataset file containing transaction data.

## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `warnings`
- `sklearn` (various modules for models, preprocessing, and metrics)
- `imbalanced-learn` (for oversampling)
- `xgboost` (if used, based on the conversation)
- `streamlit` for model deployment

## Usage
1. Create a Virtual env to run `ML.py`. 
2. Install the required libraries using `pip install -r requirements.txt`.
3. Ensure the dataset file (`creditcard.csv`) is in the same directory.
4. Run the `ML.py` script to train and evaluate machine learning models.

## Machine Learning Models

The following machine learning models were implemented and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- Linear Support Vector Machine (SVM)
- Support Vector Machine with Polynomial kernel and rbf kernel.

## Results

Evaluation metrics such as precision, recall, F1-score, and accuracy were computed for each model. Model performance varies, and it's crucial to consider business requirements when choosing the best model.

## Future Improvements

- Hyperparameter tuning for models.
- Feature engineering to enhance model performance.
- Exploration of other algorithms and ensemble methods.

## Notes

- The dataset is imbalanced, impacting model performance.
- Consider the trade-off between precision and recall based on business needs.
- Experiment with different approaches to handle class imbalance.
