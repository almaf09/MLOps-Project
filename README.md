# Customer Churn Prediction Project

## Overview
This project predicts customer churn for a telecommunications company using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. The goal is to identify customers who are likely to leave the service based on various features such as contract type, payment method, and usage metrics.

## Project Structure
- **data_preprocessing.py**: Handles data loading, preprocessing, and exploratory data analysis (EDA). It converts categorical variables, scales numerical features, and generates visualizations to understand the dataset.
- **model_building.py**: Trains and evaluates multiple models (Logistic Regression, Random Forest, KNN, and SVM) using both default parameters and hyperparameter optimization. It logs model metrics and confusion matrices to MLflow.
- **monitor_model_performance.py**: Monitors the performance of the best deployed model (Optimized Logistic Regression) on new data, calculating metrics such as accuracy, precision, recall, and F1-score, and logging them to MLflow.
- **models/**: Directory containing individual model scripts for each algorithm (default and optimized versions).
- **processed_data.csv**: The preprocessed dataset used for model training.
- **new_data.csv**: A sample of new data used for monitoring model performance.

## Dataset
The dataset includes various features such as:
- Customer demographics (gender, age)
- Contract details (contract type, payment method)
- Service usage (tenure, monthly charges, total charges)
- Target variable: Churn (whether the customer left the service)

## Workflow
1. **Data Preprocessing**: Run `data_preprocessing.py` to preprocess the raw data and perform EDA.
2. **Model Building**: Execute `model_building.py` to train and evaluate models, logging results to MLflow.
3. **Model Monitoring**: Use `monitor_model_performance.py` to evaluate the model on new data and log performance metrics.

## Requirements
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, mlflow

## Usage
- To preprocess data: `python data_preprocessing.py`
- To build and evaluate models: `python model_building.py`
- To monitor model performance: `python monitor_model_performance.py`

## Results
The project logs model performance metrics (accuracy, precision, recall, F1-score) and confusion matrices to MLflow, allowing for easy tracking and visualization of model performance over time.

## Running the Full Project

### Prerequisites
- Ensure you have Python 3.8+ installed.
- Install the required packages using:
  ```bash
  pip install -r requirements.txt
  ```

### MLflow Setup
1. **Start MLflow Tracking Server**:
   - Open a terminal and run:
     ```bash
     mlflow server --host 0.0.0.0 --port 5000
     ```
   - This will start the MLflow tracking server, allowing you to log experiments and models.

### Project Execution
1. **Data Preprocessing**:
   - Run the data preprocessing script to prepare the dataset:
     ```bash
     python data_preprocessing.py
     ```
   - This will generate `processed_data.csv` and various visualizations.

2. **Model Building**:
   - Execute the model building script to train and evaluate models:
     ```bash
     python model_building.py
     ```
   - This will log model metrics and confusion matrices to MLflow.

3. **Model Monitoring**:
   - Use the monitoring script to evaluate the model on new data:
     ```bash
     python monitor_model_performance.py
     ```
   - This will log performance metrics to MLflow.

### Viewing Results
- Navigate to `http://localhost:5000` to access the MLflow UI.
- You can view logged metrics, parameters, and artifacts (like confusion matrices) for each run.

