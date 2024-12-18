# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Function to process the dataset and train a Bayesian Linear Regression model
def process_and_train_bayesian(file_path):
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Handle missing values by imputing mean for numerical features
    dataset.fillna(dataset.mean(), inplace=True)

    # Features and target variable
    X = dataset.drop('Exam_Score', axis=1)
    y = dataset['Exam_Score']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Bayesian Ridge model
    bayesian = BayesianRidge()
    bayesian.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = bayesian.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2, y_test, y_pred

# Function to remove outliers using IQR
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# File paths
original_file = 'data/student_performance_data.csv'
outliers_removed_file = 'data/student_performance_data_outliers_are_removed_by_hand.csv'

# Preprocess the original data with the IQR method
iqr_data = pd.read_csv(original_file)
iqr_data = remove_outliers_iqr(iqr_data)
iqr_file = 'data/student_performance_data_iqr_removed.csv'
iqr_data.to_csv(iqr_file, index=False)

# Train and evaluate on all datasets using Bayesian Linear Regression
original_mse, original_mae, original_r2, original_y_test, original_y_pred = process_and_train_bayesian(original_file)
outliers_mse, outliers_mae, outliers_r2, outliers_y_test, outliers_y_pred = process_and_train_bayesian(outliers_removed_file)
iqr_mse, iqr_mae, iqr_r2, iqr_y_test, iqr_y_pred = process_and_train_bayesian(iqr_file)

# Create a comparison DataFrame for metrics
results = pd.DataFrame({
    "Dataset": ["Original Data", "Outliers Removed", "IQR Removed"],
    "Test MSE": [original_mse, outliers_mse, iqr_mse],
    "Test MAE": [original_mae, outliers_mae, iqr_mae],
    "Test R²": [original_r2, outliers_r2, iqr_r2]
})
print("Comparison of Metrics:")
print(results)

# Visualize the metrics
metrics = ["Test MSE", "Test MAE", "Test R²"]
colors = ['skyblue', 'lightgreen', 'lightcoral']

# Plot MSE and MAE in one plot
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(results))

# Plot MSE and MAE side by side for each dataset
bars_mse = plt.bar(index, results["Test MSE"], bar_width, label="MSE", color='skyblue', edgecolor='black')
bars_mae = plt.bar(index + bar_width, results["Test MAE"], bar_width, label="MAE", color='lightgreen', edgecolor='black')

# Add labels and title
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Comparison of MSE and MAE Across Datasets', fontsize=14)
plt.xticks(index + bar_width / 2, results["Dataset"], fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars for MSE and MAE
for bar in bars_mse:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

for bar in bars_mae:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Plot R² in a separate figure
plt.figure(figsize=(8, 5))
bars_r2 = plt.bar(results["Dataset"], results["Test R²"], color=['skyblue', 'lightgreen', 'lightcoral'], edgecolor='black')
for bar, value in zip(bars_r2, results["Test R²"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}",
             ha='center', va='bottom', fontsize=12)
plt.title("R² Comparison Across Datasets", fontsize=14)
plt.ylabel("R² Score", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Scatter plots for predictions
datasets = [
    ("Original Data", original_y_test, original_y_pred),
    ("Outliers Removed", outliers_y_test, outliers_y_pred),
    ("IQR Removed", iqr_y_test, iqr_y_pred)
]

for name, y_test, y_pred in datasets:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal Prediction (y=x)')
    plt.xlabel('Actual Exam Score', fontsize=12)
    plt.ylabel('Predicted Exam Score', fontsize=12)
    plt.title(f'Predicted vs Actual Exam Scores on Test Set ({name})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
