import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
RANDOM FOREST - DEFAULT
""")
    df = pd.read_csv('processed_data.csv')
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run(run_name='Random Forest - Default'):
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_param('model', 'Random Forest')
        mlflow.log_param('n_estimators', 100)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        plot_and_log_confusion_matrix(y_test, y_pred, 'random_forest_default')
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_artifact('processed_data.csv')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

if __name__ == '__main__':
    main() 