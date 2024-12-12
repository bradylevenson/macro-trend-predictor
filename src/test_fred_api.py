import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Load the FRED API key from the .env file
load_dotenv()

# Initialize FRED API with your key
api_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key=api_key)

# Fetch GDP, Unemployment, and Inflation data from FRED
gdp = fred.get_series('GDP')
unemployment = fred.get_series('UNRATE')
inflation = fred.get_series('CPIAUCSL')

# Combine the data into a single DataFrame
data = pd.DataFrame({
    'GDP': gdp,
    'Unemployment': unemployment,
    'Inflation': inflation
})

# Handle missing and infinite values by replacing them with the mean
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
data.fillna(data.mean(), inplace=True)  # Replace NaN values with the column mean

# Preprocess the data
data['GDP_Growth'] = data['GDP'].pct_change() * 100  # Percentage change in GDP (growth rate)
data['Unemployment_Change'] = data['Unemployment'].pct_change() * 100
data['Inflation_Rate'] = data['Inflation'].pct_change() * 100
data = data.dropna()  # Drop rows where any of the columns is NaN after calculations

# Create a new column to categorize GDP growth (target variable)
def categorize_growth(growth):
    if growth >= 2:
        return 'High'
    else:
        return 'Low'

data['GDP_Growth_Label'] = data['GDP_Growth'].apply(categorize_growth)

# Prepare features and target variable
X = data[['GDP', 'Unemployment', 'Inflation', 'Unemployment_Change', 'Inflation_Rate']]
y = data['GDP_Growth_Label']

# Encode the target variable ('High', 'Low' -> 0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Best hyperparameters found by GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Use the best model to make predictions
best_rf_classifier = grid_search.best_estimator_
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot feature importances
importances = best_rf_classifier.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

# Compute precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate and visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))
plt.show()

# Plot Precision-Recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.join(os.path.dirname(__file__), 'precision_recall_curve.png'))
plt.show()

# Save evaluation metrics
metrics = {
    'Accuracy': accuracy,
    'Precision': precision.mean(),
    'Recall': recall.mean(),
    'F1 Score': f1
}

save_dir = os.path.dirname(__file__)
model_save_path = os.path.join(save_dir, 'rf_model.pkl')
encoder_save_path = os.path.join(save_dir, 'label_encoder.pkl')
metrics_path = os.path.join(save_dir, 'evaluation_metrics.pkl')

with open(model_save_path, 'wb') as f:
    pickle.dump(best_rf_classifier, f)

with open(encoder_save_path, 'wb') as f:
    pickle.dump(label_encoder, f)

with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)

print("Model, label encoder, and evaluation metrics saved!")
