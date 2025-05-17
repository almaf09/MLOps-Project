import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the preprocessed data"""
    print("Loading preprocessed data...")
    df = pd.read_csv('processed_data.csv')
    
    # Drop non-feature columns
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
    
    return df

def prepare_data(df):
    """Prepare features and target variable"""
    print("\nPreparing features and target...")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def plot_and_log_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    fname = f'{model_name}_confusion_matrix.png'
    plt.savefig(fname)
    plt.close()
    mlflow.log_artifact(fname)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models, log everything to MLflow"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(random_state=42, kernel='rbf', probability=True)
    }
    params = {
        'Logistic Regression': {'max_iter': 500},
        'Random Forest': {'n_estimators': 100},
        'KNN': {'n_neighbors': 5},
        'SVM': {'kernel': 'rbf'}
    }
    grids = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200]},
        'KNN': {'n_neighbors': [3, 5, 7]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    }
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name} - Default"):
            print(f"\nTraining {name} (Default)...")
            if name == 'KNN':
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_param("model", name)
            for k, v in params[name].items():
                mlflow.log_param(k, v)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            plot_and_log_confusion_matrix(y_test, y_pred, f"{name.replace(' ', '_')}_default")
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact('processed_data.csv')
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        # Run with hyperparameter optimization
        with mlflow.start_run(run_name=f"{name} - Optimized"):
            print(f"\nTraining {name} (Optimized)...")
            grid = GridSearchCV(model, grids[name], cv=3, scoring='accuracy')
            if name == 'KNN':
                grid.fit(X_train.values, y_train.values)
                y_pred = grid.predict(X_test.values)
            else:
                grid.fit(X_train, y_train)
                y_pred = grid.predict(X_test)
            best_model = grid.best_estimator_
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_param("model", name)
            for k, v in grid.best_params_.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            plot_and_log_confusion_matrix(y_test, y_pred, f"{name.replace(' ', '_')}_optimized")
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_artifact('processed_data.csv')
            print(f"Best Params: {grid.best_params_}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 