import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    model_name = "Telco_CustomerChurn"
    model_version = 1  
    model_uri = f"models:/{model_name}/{model_version}"
    new_data_path = "new_data.csv"  

    print(f"Loading model {model_name} version {model_version} from MLflow registry...")
    model = mlflow.sklearn.load_model(model_uri)

    print(f"Loading new data from {new_data_path}...")
    new_data = pd.read_csv(new_data_path)
    if 'customerID' in new_data.columns:
        new_data = new_data.drop('customerID', axis=1)
    before_drop = len(new_data)
    new_data = new_data.dropna()
    after_drop = len(new_data)
    if before_drop != after_drop:
        print(f"Warning: Dropped {before_drop - after_drop} rows with missing values.")
    X_new = new_data.drop("Churn", axis=1)
    y_true = new_data["Churn"]

    print("Making predictions and calculating metrics...")
    y_pred = model.predict(X_new)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("Logging metrics and confusion matrix to MLflow...")
    with mlflow.start_run(run_name="Performance Monitoring"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix_monitoring.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix_monitoring.png")
        print(f"Logged monitoring metrics for {model_name} v{model_version}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main() 