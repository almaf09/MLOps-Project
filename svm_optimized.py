import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

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

def main():
    print("""
SVM - OPTIMIZED
""")
    df = pd.read_csv('processed_data.csv')
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run(run_name='SVM - Optimized'):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        grid = GridSearchCV(SVC(random_state=42, probability=True), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_param('model', 'SVM')
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        plot_and_log_confusion_matrix(y_test, y_pred, 'svm_optimized')
        mlflow.sklearn.log_model(best_model, 'model')
        mlflow.log_artifact('processed_data.csv')
        print(f'Best Params: {grid.best_params_}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

if __name__ == '__main__':
    main() 