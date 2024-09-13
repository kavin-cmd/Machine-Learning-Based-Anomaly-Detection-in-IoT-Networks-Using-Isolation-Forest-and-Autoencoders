from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from preprocessing import load_and_preprocess_data
import os

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def load_config():
    with open('config.json') as f:
        return json.load(f)

def train_isolation_forest(X_train, y_train, X_test, y_test):
    """
    Train Isolation Forest model and evaluate it.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing labels.

    Returns:
    - None
    """
    config = load_config()
    contamination = config["isolation_forest"]["contamination"]

    # Initialize Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model
    model.fit(X_train)

    # Predict anomalies on test data
    y_pred = model.predict(X_test)
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]  # Convert -1 to 1 (anomaly) and 1 to 0 (normal)

    # Evaluate the model
    print("Isolation Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    config = load_config()
    model_path = config["models_path"] + 'isolation_forest_model.pkl'
    joblib.dump(model, model_path)

if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset_path"]
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    # Train and evaluate Isolation Forest model
    train_isolation_forest(X_train, y_train, X_test, y_test)
