"""
Wine Quality Classification Model

This script analyzes a wine quality dataset and builds classification models to predict
whether a wine is good (quality >= 6) or bad (quality < 6) based on its properties.

The script:
1. Loads and explores the wine dataset
2. Preprocesses the data, converting quality scores to binary classification
3. Trains three machine learning models: Logistic Regression, Random Forest, and SVM
4. Evaluates and compares model performance using various metrics

Dependencies:
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scikit-learn: Machine learning algorithms and utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------
# Data Loading and Exploration
# -------------------------------------------

# Load the wine quality dataset
file_path = 'wine-quality-white-and-red.csv'
wine_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Preview of the dataset:")
print(wine_data.head())

# Display information about the dataset (columns, data types, non-null values)
print("\nDataset Information:")
print(wine_data.info())

# Check for missing values in the dataset
missing_values = wine_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# -------------------------------------------
# Data Preprocessing
# -------------------------------------------

# Convert categorical column 'type' to numerical using one-hot encoding (if present)
if 'type' in wine_data.columns:
    print("\nConverting wine type to numerical values")
    wine_data = pd.get_dummies(wine_data, columns=['type'], drop_first=True)
    # Note: drop_first=True avoids multicollinearity by removing one redundant column

# Separate features (X) and target variable (y)
X = wine_data.drop(columns=['quality'])  # All columns except 'quality' are features
y = wine_data['quality']                 # 'quality' is the target variable

# Convert quality rating into binary classification:
# Good quality (1): rating >= 6
# Bad quality (0): rating < 6
y = np.where(y >= 6, 1, 0)
print(f"\nClass distribution after binarization: {np.bincount(y)}")
print(f"Positive class (Good wine) percentage: {np.mean(y) * 100:.2f}%")

# Split the dataset into training (80%) and testing (20%) sets
# random_state ensures reproducibility, stratify maintains class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Standardize features: rescaling to mean=0 and variance=1
# This is important for models like SVM and Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform it
X_test = scaler.transform(X_test)        # Apply same transformation to test data

# -------------------------------------------
# Model Training and Evaluation
# -------------------------------------------

# Initialize the classification models to test
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='linear')
}

# Dictionary to store accuracy results for comparison
results = {}

# Train each model and evaluate performance
for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"Training and evaluating: {name}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and store in results dictionary
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    # Print performance metrics
    print(f"\n{name} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Display detailed classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and display confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# -------------------------------------------
# Model Comparison
# -------------------------------------------

# Create a bar chart to compare the performance of all models
plt.figure(figsize=(10, 6))
bars = plt.bar(
    results.keys(), 
    results.values(), 
    color=['#3498db', '#2ecc71', '#e74c3c']
)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.01,
        f'{height:.4f}',
        ha='center', 
        va='bottom'
    )

plt.xlabel("Classification Model")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Wine Quality Classification Models")
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1 (accuracy range)
plt.xticks(rotation=15)  # Rotate labels for better readability
plt.tight_layout()
plt.show()

print("\nAnalysis complete!")