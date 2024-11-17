# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

# Random Forest Regressor with K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
random_forest = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],       # Number of trees in the forest
    'max_depth': [None, 10, 20],          # Depth of each tree
    'min_samples_split': [2, 5, 10],      # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]         # Minimum samples required to be at a leaf node
}

grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf,
                              scoring='r2', cv=5, verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = grid_search_rf.best_estimator_

# R² score on the test set
rf_test_r2 = r2_score(y_test, best_rf_model.predict(X_test))

# Test accuracy within tolerance for Random Forest
tolerance = 0.6  # Define the tolerance range for accuracy
rf_accuracy = calculate_accuracy(y_test, best_rf_model.predict(X_test), tolerance)

# Prepare results for display
results = {
    "Model": ["Random Forest Regression"],
    "Mean CV R^2": [grid_search_rf.best_score_],
    "Test R^2": [rf_test_r2],
    "Best Parameters": [grid_search_rf.best_params_],
    "Accuracy (%)": [rf_accuracy]
}

results_df = pd.DataFrame(results)

# Print the results DataFrame
print("Random Forest Performance Results:")
print(results_df)

# Save results to a CSV file for further inspection
results_df.to_csv("../output/random_forest_performance_with_accuracy.csv", index=False)

# Compare predictions for Random Forest
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Random Forest Regression": best_rf_model.predict(X_test),
})

# Save the comparison to a CSV file
comparison_df.to_csv("../output/random_forest_predictions_comparison.csv", index=False)
print("Predictions saved to ../output/random_forest_predictions_comparison.csv")

# Visualize Test R^2 Scores
models = results_df["Model"]
test_r2_scores = results_df["Test R^2"]

plt.figure(figsize=(8, 5))
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

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracy_scores, color='lightgreen', edgecolor='black')

# Adding labels on top of each bar
for bar, score in zip(bars, accuracy_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{score:.2f}%", ha='center', va='bottom', fontsize=12)

plt.title("Accuracy (%) of Models (Tolerance ±0.6)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.ylim(0, 100)  # Accuracy ranges from 0% to 100%
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
