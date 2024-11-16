# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Function to calculate accuracy within a tolerance range
def calculate_accuracy(actual, predicted, tolerance):
    """
    Calculate the accuracy of predictions within a given tolerance range.

    Parameters:
        actual (array-like): True values.
        predicted (array-like): Predicted values.
        tolerance (float): Acceptable deviation from true values.

    Returns:
        float: Accuracy as a percentage.
    """
    accurate_predictions = np.abs(actual - predicted) <= tolerance
    accuracy = np.mean(accurate_predictions) * 100  # Convert to percentage
    return accuracy

# Load the dataset
file_path = '../data/student_performance_data.csv'  # Replace with the correct path
dataset = pd.read_csv(file_path)

# Handle missing values by imputing mean for numerical features
dataset.fillna(dataset.mean(), inplace=True)

# Features and target variable
X = dataset.drop('Exam_Score', axis=1)
y = dataset['Exam_Score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression with K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linear_model = LinearRegression()
cv_r2_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
    linear_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = linear_model.predict(X_val_cv)
    cv_r2_scores.append(r2_score(y_val_cv, y_pred_cv))

mean_r2_linear = np.mean(cv_r2_scores)

# Ridge Regularization
ridge = Ridge()
param_grid_ridge = {'alpha': np.logspace(-4, 4, 50)}
ridge_search = GridSearchCV(ridge, param_grid_ridge, scoring='r2', cv=5)
ridge_search.fit(X_train, y_train)
best_ridge_model = ridge_search.best_estimator_
ridge_test_r2 = r2_score(y_test, best_ridge_model.predict(X_test))

# Lasso Regularization
lasso = Lasso(max_iter=10000)
param_grid_lasso = {'alpha': np.logspace(-4, 4, 50)}
lasso_search = GridSearchCV(lasso, param_grid_lasso, scoring='r2', cv=5)
lasso_search.fit(X_train, y_train)
best_lasso_model = lasso_search.best_estimator_
lasso_test_r2 = r2_score(y_test, best_lasso_model.predict(X_test))

# Test accuracy for Linear Regression
linear_test_r2 = r2_score(y_test, linear_model.predict(X_test))

# Accuracy within tolerance for all models
tolerance = 0.6 # Define the tolerance range for accuracy
linear_accuracy = calculate_accuracy(y_test, linear_model.predict(X_test), tolerance)
ridge_accuracy = calculate_accuracy(y_test, best_ridge_model.predict(X_test), tolerance)
lasso_accuracy = calculate_accuracy(y_test, best_lasso_model.predict(X_test), tolerance)

# Prepare results for display
results = {
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "Mean CV R^2": [mean_r2_linear, ridge_search.best_score_, lasso_search.best_score_],
    "Test R^2": [linear_test_r2, ridge_test_r2, lasso_test_r2],
    "Best Alpha": [None, ridge_search.best_params_['alpha'], lasso_search.best_params_['alpha']],
    "Accuracy (%)": [linear_accuracy, ridge_accuracy, lasso_accuracy]
}

results_df = pd.DataFrame(results)

# Print the results DataFrame
print("Model Performance Results:")
print(results_df)

# Save results to a CSV file for further inspection
results_df.to_csv("../output/model_performance_with_accuracy.csv", index=False)

# Compare predictions for all models
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Linear Regression": linear_model.predict(X_test),
    "Ridge Regression": best_ridge_model.predict(X_test),
    "Lasso Regression": best_lasso_model.predict(X_test),
})

# Save the comparison to a CSV file
comparison_df.to_csv("../output/model_predictions_comparison.csv", index=False)
print("Predictions saved to ../output/model_predictions_comparison.csv")

# Visualize Test R^2 Scores
models = results_df["Model"]
test_r2_scores = results_df["Test R^2"]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, test_r2_scores, color='skyblue', edgecolor='black')

# Adding labels on top of each bar
for bar, score in zip(bars, test_r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{score:.4f}", ha='center', va='bottom', fontsize=12)

plt.title("Test R² Scores of Models", fontsize=14)
plt.ylabel("R² Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.ylim(0, 1)  # R² score typically ranges from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualize Accuracy
accuracy_scores = results_df["Accuracy (%)"]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy_scores, color='lightgreen', edgecolor='black')

# Adding labels on top of each bar
for bar, score in zip(bars, accuracy_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{score:.4f}%", ha='center', va='bottom', fontsize=12)

plt.title("Accuracy (%) of Models (Tolerance ±0.5)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.ylim(0, 100)  # Accuracy ranges from 0% to 100%
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
