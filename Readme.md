Machine Learning-Based Anomaly Detection in IoT Networks

Project Overview

This project focuses on developing machine learning models to detect anomalies in IoT network traffic. By leveraging Isolation Forest and Autoencoder techniques, the goal is to identify potential security threats and operational irregularities within IoT networks. This project emphasizes the preprocessing of high-dimensional data, feature extraction, model training, and evaluation, contributing to the overall reliability and security of IoT systems.

Project Structure

├── data/
│   ├── raw/                   # Raw IoT network traffic data
│   ├── processed/             # Processed data after cleaning and feature extraction
│   └── README.md              # Description of the data and sources
├── notebooks/
│   ├── data_preprocessing.ipynb  # Data cleaning and preprocessing steps
│   ├── isolation_forest.ipynb    # Implementation of Isolation Forest for anomaly detection
│   ├── autoencoder.ipynb         # Autoencoder model training and evaluation
│   └── analysis.ipynb            # Comparative analysis and visualization of results
├── scripts/
│   ├── preprocess_data.py        # Script for data preprocessing
│   ├── train_isolation_forest.py # Script to train and evaluate the Isolation Forest model
│   └── train_autoencoder.py      # Script to train and evaluate the Autoencoder model
├── results/
│   ├── model_performance/        # Model performance metrics and evaluation results
│   ├── visualizations/           # Plots and visual representations of the data and results
│   └── README.md                 # Description of the results and visualizations
└── README.md                     # Project overview and instructions
Getting Started

Prerequisites

Python 3.8 or higher

Required libraries:
numpy
pandas
scikit-learn
tensorflow or keras
matplotlib
seaborn
jupyter (optional, for running notebooks)
You can install all the required libraries using:
pip install -r requirements.txt

Data Collection

The dataset used in this project can be sourced from publicly available IoT network traffic datasets like UNSW-NB15 or TON_IoT.

Place the raw data in the data/raw/ directory.

Data Preprocessing
Run the data_preprocessing.ipynb notebook or the preprocess_data.py script to clean and preprocess the data.

The processed data will be saved in the data/processed/ directory.

Model Training

Isolation Forest:
Train the Isolation Forest model by running the isolation_forest.ipynb notebook or the train_isolation_forest.py script.

The trained model and performance metrics will be stored in the results/model_performance/ directory.

Autoencoder:
Train the Autoencoder model by running the autoencoder.ipynb notebook or the train_autoencoder.py script.

The trained model and performance metrics will be stored in the results/model_performance/ directory.

Model Evaluation and Analysis

Compare the performance of the two models by running the analysis.ipynb notebook.

Visualizations and analysis results will be saved in the results/visualizations/ directory.

Results

The project resulted in significant improvements in anomaly detection accuracy using both models.

The comparative analysis highlighted the strengths and weaknesses of each approach, providing insights into their applicability in different IoT network environments.

Contributing

If you wish to contribute to this project, feel free to fork the repository and submit a pull request with your improvements.

License

This project is licensed under the Apache License - see the LICENSE file for details.

Contact

For any inquiries or suggestions, please contact Kavin Parikh.