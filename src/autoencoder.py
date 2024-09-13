import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import load_and_preprocess_data
import json

def load_config():
    with open('config.json') as f:
        return json.load(f)

def build_autoencoder(input_shape):
    """
    Build and compile the Autoencoder model.

    Parameters:
    - input_shape (int): Number of features in the dataset.

    Returns:
    - autoencoder (tf.keras.Model): Compiled Autoencoder model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(input_shape, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(X_train, X_test):
    """
    Train Autoencoder model and evaluate it.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.

    Returns:
    - None
    """
    config = load_config()
    epochs = config["autoencoder"]["epochs"]
    batch_size = config["autoencoder"]["batch_size"]
    validation_split = config["autoencoder"]["validation_split"]

    input_shape = X_train.shape[1]
    autoencoder = build_autoencoder(input_shape)

    # Train the model
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    # Reconstruct the test data
    X_test_reconstructed = autoencoder.predict(X_test)

    # Calculate reconstruction error
    reconstruction_error = np.mean(np.abs(X_test - X_test_reconstructed), axis=1)
    threshold = np.percentile(reconstruction_error, 99)  # Set threshold for anomaly detection

    # Predict anomalies
    y_pred = [1 if error > threshold else 0 for error in reconstruction_error]

    # Load true labels for evaluation
    print("Autoencoder Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    model_path = config["models_path"] + 'autoencoder_model.h5'
    autoencoder.save(model_path)

if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset_path"]
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    # Train and evaluate Autoencoder model
    train_autoencoder(X_train, X_test)
