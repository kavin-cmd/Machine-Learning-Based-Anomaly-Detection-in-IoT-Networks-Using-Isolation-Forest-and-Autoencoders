import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import json

def load_config():
    with open('config.json') as f:
        return json.load(f)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the KDD Cup 99 dataset.

    Parameters:
    - file_path (str): Path to the dataset file.

    Returns:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Column names for the dataset
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
               "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "lnum_compromised",
               "lroot_shell", "lsu_attempted", "lnum_root", "lnum_file_creations", "lnum_shells",
               "lnum_access_files", "lnum_outbound_cmds", "is_host_login", "is_guest_login", "count",
               "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
               "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
               "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    df.columns = columns

    # Handle missing values if any
    df = df.fillna(0)  # Replace missing values with 0, adjust strategy as needed

    # Encode categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

    # Convert 'label' to binary values: 0 for 'normal', 1 for anomalies
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Normalize features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset_path"]
    
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
    print("Data preprocessing completed.")
