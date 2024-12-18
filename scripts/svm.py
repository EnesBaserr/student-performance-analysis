# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
file_path = '../data/student_performance_data.csv'  # Replace with your dataset path
dataset = pd.read_csv(file_path)

# Handle missing values by imputing the mean
dataset.fillna(dataset.mean(), inplace=True)

# Define features (X) and target variable (y)
X = dataset.drop('Exam_Score', axis=1)  # Replace 'Exam_Score' with the actual target column name
y = dataset['Exam_Score']

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Further reduce the sample size for training
# Reduce training data to 30% of the original training set for testing purposes
X_train_sample = X_train.sample(frac=0.3, random_state=42)  # 30% of the training data
y_train_sample = y_train[X_train_sample.index]

# Step 4: Train and evaluate SVR models with different C values
c_values = [10, 100]  # Specify the C values to test
results = []

for c in c_values:
    # Initialize and train the SVR model with the current C value
    svr_model = SVR(kernel='linear', C=c, epsilon=0.2)  # Use epsilon=0.2
    svr_model.fit(X_train_sample, y_train_sample)

    # Evaluate the model on the test set
    test_r2 = r2_score(y_test, svr_model.predict(X_test))
    results.append({"C": c, "Test R^2": test_r2})
    print(f"C={c}, Test R^2={test_r2:.4f}")

# Step 5: Create a DataFrame to display results
results_df = pd.DataFrame(results)

# Print the results for all C values
print("\nPerformance Comparison:")
print(results_df)

# Step 6: Plot the comparison
plt.figure(figsize=(8, 5))
plt.plot(results_df["C"], results_df["Test R^2"], marker='o', linestyle='--', label='Test R^2')
plt.title("SVR Performance for Different C Values (Reduced Sample Size)")
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Test R^2 Score")
plt.xscale('log')  # Log scale for C values
plt.grid(True)
plt.legend()
plt.show()
