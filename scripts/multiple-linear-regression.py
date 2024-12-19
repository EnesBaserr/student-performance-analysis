import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ridge Regression Function
def ridge_regression(data, dataset_name):
    X = data.drop('Exam_Score', axis=1)
    y = data['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ridge Regularization
    ridge = Ridge()
    param_grid_ridge = {'alpha': np.logspace(-4, 4, 50)}
    ridge_search = GridSearchCV(ridge, param_grid_ridge, scoring='r2', cv=5)
    ridge_search.fit(X_train, y_train)
    best_ridge_model = ridge_search.best_estimator_

    y_pred = best_ridge_model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot Predicted vs Actual Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
    plt.title(f"Ridge Regression - {dataset_name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    # Add metrics as labels
    plt.text(0.50, 0.95, f"R²: {r2:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.show()

    return mse, mae, r2


# Linear Regression Function
def linear_regression(data, dataset_name):
    X = data.drop('Exam_Score', axis=1)
    y = data['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_pred = linear_model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot Predicted vs Actual Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
    plt.title(f"Linear Regression - {dataset_name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    # Add metrics as labels
    plt.text(0.50, 0.95, f"R²: {r2:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.show()


    return mse, mae, r2

# Load and preprocess data
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)
    return data

# Evaluate Ridge and Linear Regression
file_paths = [
    '../data/student_performance_data_outliers_are_removed_by_hand.csv',
    '../data/student_performance_data.csv'
]

for i,file_path in enumerate (file_paths):
    dataset_name= "Outliers Removed Dataset" if i==0 else "Original Dataset"
    data = preprocess_data(file_path)

    print(f"\nResults for {dataset_name}:")

    # Ridge Regression
    ridge_mse, ridge_mae, ridge_r2 = ridge_regression(data, dataset_name)

    # Linear Regression
    linear_mse, linear_mae, linear_r2 = linear_regression(data, dataset_name)
