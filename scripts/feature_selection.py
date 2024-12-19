import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import shap


# Load and preprocess data
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)
    return data


# Function to find important features
def feature_importance_random_forest(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    # Feature importance
    feature_importances = rf.feature_importances_
    features = X.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importances using Random Forest")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()

    return importance_df, rf, X, y


# Function to visualize correlation matrix
def visualize_correlation_matrix(data):
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()


# Function to plot relationships between top features and target with trend curves
def visualize_top_features_vs_target(data, target_column, top_features):
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=feature, y=target_column, alpha=0.6, label="Data Points")

        # Add trend curve using regression
        sns.regplot(data=data, x=feature, y=target_column, scatter=False, ci=None, color='red', label="Trend Curve")

        plt.title(f"Relationship between {feature} and {target_column} with Trend Curve")
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.legend()
        plt.grid()
        plt.show()


# Function to calculate and visualize SHAP values
def calculate_shap_values(model, X, top_n=5):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary plot showing global feature importance
    print("\nGenerating SHAP summary plot for global feature importance...")
    shap.summary_plot(shap_values, X)

    # Dependence plot for top features
    # top_features = X.columns[:top_n]
    # for feature in top_features:
    #     print(f"\nGenerating SHAP dependence plot for feature: {feature}")
    #     shap.dependence_plot(feature, shap_values.values, X)


# Load the dataset
file_path = '../data/student_performance_data_outliers_are_removed_by_hand.csv'
data = preprocess_data(file_path)

# Find and plot important features
important_features_df, rf, X, y = feature_importance_random_forest(data, target_column='Exam_Score')

# Display the top features
print("Top Important Features:")
print(important_features_df.head(10))

# Visualize the correlation matrix
visualize_correlation_matrix(data)

# Visualize top 5 features vs target with trend curve
top_5_features = important_features_df["Feature"].head(5).tolist()
visualize_top_features_vs_target(data, target_column='Exam_Score', top_features=top_5_features)

# Calculate and visualize SHAP values for the top features
calculate_shap_values(rf, X, top_n=5)