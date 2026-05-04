# Water Potability Prediction Using Machine Learning

This project predicts whether a water sample is potable or not potable using water quality measurements. It compares three machine learning algorithms and deploys the best-performing model in a Streamlit web application.

## Models Compared

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Weighted F1-score
- Confusion Matrix

## Files

- `water_potability.csv` - Dataset
- `water_potability_model_comparison.ipynb` - Notebook for preprocessing, model training, and evaluation
- `best_model.pkl` - Saved best model pipeline
- `model_metadata.json` - Model results and feature information
- `model_results.csv` - Model comparison table
- `app.py` - Streamlit application
- `requirements.txt` - Required Python packages

## Setup Instructions

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Application Usage

1. Enter water quality values in the sidebar.
2. Click **Predict Potability**.
3. View the prediction and probability output.
4. Review the model performance summary shown in the app.

PTF03 FINAL PROJECT
