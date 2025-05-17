import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Convert binary categorical variables to numeric
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
    
    # One-hot encode specific categorical variables
    categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded

def plot_all_distributions(df):
    """Plot distributions for all columns"""
    print("\nCreating distribution plots for all columns...")
    
    # Get numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    
    # Plot numerical distributions
    for col in numerical_cols:
        if col != 'Churn':  
            plt.figure(figsize=(10, 6))
            # Check if the column is binary 
            unique_values = df[col].dropna().unique()
            if set(unique_values).issubset({0, 1}):
                # For binary features, use countplot
                sns.countplot(data=df, x=col)
                plt.xticks([0, 1], ['No', 'Yes'])
            else:
                # For non-binary features, use histplot
                sns.histplot(data=df, x=col, bins=30)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(f'dist_{col.lower()}.png')
            plt.close()
    
    # Plot categorical distributions
    for col in categorical_cols:
        if col != 'customerID':  
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=col)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'dist_{col.lower()}.png')
            plt.close()

def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\nPerforming exploratory data analysis...")
    
    print("\n=== Dataset Information ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n=== First Few Rows ===")
    print(df.head())
    
    print("\n=== Data Types and Non-null Counts ===")
    print(df.info())
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    print("\nCreating visualizations...")
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Churn')
    plt.title('Distribution of Churn')
    plt.savefig('churn_distribution.png')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation = df[numerical_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    plot_all_distributions(df)
    
    print("\n=== Key Findings ===")
    print(f"Churn Rate: {df['Churn'].mean():.2%}")
    
    churn_correlations = correlation['Churn'].sort_values(ascending=False)
    print("\nTop correlations with Churn:")
    print(churn_correlations)

def main():

    df = load_and_preprocess_data('Telco.csv')
    perform_eda(df)
    df.to_csv('processed_data.csv', index=False)
    print("\nProcessed data saved to 'processed_data.csv'")
    print("\nAnalysis completed! Check the generated visualization files.")

if __name__ == "__main__":
    main() 