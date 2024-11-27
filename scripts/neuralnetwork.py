# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

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
file_path = 'data/student_performance_data.csv'  # Replace with the correct path
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

# Build the Neural Network model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with validation split
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=1)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on Test Set: {mae}")

# Predictions and performance metrics
y_pred = model.predict(X_test)
nn_test_r2 = r2_score(y_test, y_pred)
tolerance = 2  # Define the tolerance range for accuracy
nn_accuracy = calculate_accuracy(y_test, y_pred.flatten(), tolerance)

# Prepare results for display
results = {
    "Model": ["Neural Network"],
    "Test R^2": [nn_test_r2],
    "Accuracy (%)": [nn_accuracy]
}
results_df = pd.DataFrame(results)

# Print the results DataFrame
print("Neural Network Performance Results:")
print(results_df)

# Save results to a CSV file for further inspection
results_df.to_csv("output/neural_network_performance_with_accuracy.csv", index=False)

# Compare predictions for Neural Network
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Neural Network": y_pred.flatten(),
})

# Save the comparison to a CSV file
comparison_df.to_csv("output/neural_network_predictions_comparison.csv", index=False)
print("Predictions saved to output/neural_network_predictions_comparison.csv")

# Visualize Test R^2 Scores
plt.figure(figsize=(8, 5))
bars = plt.bar(results_df["Model"], results_df["Test R^2"], color='skyblue', edgecolor='black')
for bar, score in zip(bars, results_df["Test R^2"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.4f}",
             ha='center', va='bottom', fontsize=12)
plt.title("Test R² Scores of Models", fontsize=14)
plt.ylabel("R² Score", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualize Accuracy
plt.figure(figsize=(8, 5))
bars = plt.bar(results_df["Model"], results_df["Accuracy (%)"], color='lightgreen', edgecolor='black')
for bar, score in zip(bars, results_df["Accuracy (%)"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.2f}%",
             ha='center', va='bottom', fontsize=12)
plt.title("Accuracy (%) of Models (Tolerance ±2)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Scatter plot: Actual vs Predicted Exam Scores with color coding
plt.figure(figsize=(8, 6))
differences = np.abs(y_test - y_pred.flatten())
green_points = differences <= 2
red_points = differences > 2
plt.scatter(y_test[green_points], y_pred.flatten()[green_points], color='green', alpha=0.7, label='Within ±2 Points')
plt.scatter(y_test[red_points], y_pred.flatten()[red_points], color='red', alpha=0.7, label='Beyond ±2 Points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--b', label='Ideal Prediction (y=x)')
plt.xlabel('Actual Exam Score', fontsize=12)
plt.ylabel('Predicted Exam Score', fontsize=12)
plt.title('Predicted vs Actual Exam Scores on Test Set', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Combine Accuracy and R² Score in one graph
plt.figure(figsize=(8, 6))
metrics = ['R² Score', 'Accuracy (%)']
values = [nn_test_r2 * 100, nn_accuracy]  # Scale R² score to percentage
bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen'], edgecolor='black')
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{value:.2f}", ha='center', va='bottom', fontsize=12)
plt.title('R² Score and Accuracy', fontsize=14)
plt.ylabel('Metric Value (%)', fontsize=12)
plt.ylim(0, 100)  # Set limit to accommodate percentage metrics
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot MSE Loss and MAE in one graph
plt.figure(figsize=(10, 6))
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']
plt.plot(train_loss, label='Training Loss (MSE)', color='blue', linestyle='-', linewidth=2)
plt.plot(val_loss, label='Validation Loss (MSE)', color='orange', linestyle='--', linewidth=2)
plt.plot(train_mae, label='Training MAE', color='green', linestyle='-', linewidth=2)
plt.plot(val_mae, label='Validation MAE', color='red', linestyle='--', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('MSE Loss and MAE for Each Epoch', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
