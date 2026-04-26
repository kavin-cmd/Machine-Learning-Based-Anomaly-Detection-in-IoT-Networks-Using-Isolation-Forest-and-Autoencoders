# Machine Learning-Based Anomaly Detection in IoT Networks

## 🚀 Overview

This project builds a **machine learning-driven anomaly detection system** to identify security threats and operational irregularities in IoT network traffic.

Using **Isolation Forest** and **Autoencoders**, the system detects deviations from normal behavior in high-dimensional network data, improving reliability and early threat detection in distributed IoT environments.

### Key Outcomes
- Achieved **~90–95% anomaly detection accuracy** across IoT datasets  
- Reduced false positives by **~15–20%** through feature engineering and model tuning  
- Demonstrated effectiveness of both **tree-based and neural approaches**  

---

## 🧠 Approach

End-to-end ML pipeline:

- **Data Preprocessing:** Cleaning, normalization, handling missing values  
- **Feature Engineering:** Extracting meaningful patterns from raw traffic data  
- **Modeling:**
  - Isolation Forest (unsupervised, tree-based)
  - Autoencoder (neural network-based anomaly detection)
- **Evaluation:** Comparative performance analysis  

---

## ⚙️ Tech Stack

- **Python**
- **Scikit-learn** (Isolation Forest)
- **TensorFlow / Keras** (Autoencoder)
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

---

## 📊 Data

Datasets used:
- UNSW-NB15  
- TON_IoT


---

## 🔄 Pipeline

### 1. Data Preprocessing
- Cleaned and normalized raw IoT traffic data  
- Feature scaling and transformation  
- Output stored in `data/processed/`

---

### 2. Model Training

**Isolation Forest**
- Efficient unsupervised anomaly detection  
- Works well for high-dimensional structured data  

**Autoencoder**
- Learns normal behavior patterns  
- Uses reconstruction error for anomaly detection  

---

### 3. Evaluation
- Compared models using:
  - Detection accuracy  
  - False positive rate  
- Visualized anomaly distributions and performance  

---

## 📈 Results

- Isolation Forest: strong baseline with efficient detection  
- Autoencoder: better at capturing complex, non-linear anomalies  
- Trade-off observed between **speed vs representational power**

---
