# Heart Disease Prediction Model

This repository contains a machine learning model that predicts heart disease using the UCI Heart Disease dataset. The model was trained using **K-Nearest Neighbors (KNN)** and **Decision Tree** algorithms, and a **Streamlit** app is provided for interactive web deployment.

## 🧑‍⚕️ Dataset Description

### Heart Disease UCI Dataset – Overview

This dataset is a cleaned version of the original Cleveland Heart Disease dataset from the UCI Machine Learning Repository, made available via Kaggle.

**Link to dataset:** [Kaggle Heart Disease UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

### 📁 Features:

| Feature         | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `age`           | Age of the patient (in years)                                               |
| `sex`           | Gender (1 = male, 0 = female)                                               |
| `cp`            | Chest pain type (0–3)                                                       |
| `trestbps`      | Resting blood pressure (in mm Hg)                                           |
| `chol`          | Serum cholesterol in mg/dl                                                  |
| `fbs`           | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)                       |
| `restecg`       | Resting electrocardiographic results (values 0, 1, 2)                       |
| `thalach`       | Maximum heart rate achieved                                                 |
| `exang`         | Exercise-induced angina (1 = yes; 0 = no)                                   |
| `oldpeak`       | ST depression induced by exercise relative to rest                          |
| `slope`         | Slope of the peak exercise ST segment (0–2)                                 |
| `ca`            | Number of major vessels (0–3) colored by fluoroscopy                        |
| `thal`          | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)           |
| `target`        | Diagnosis of heart disease (1 = disease; 0 = no disease)                    |

---

## 🚀 Model Description

The project includes the following machine learning models:

- **K-Nearest Neighbors (KNN)**: A supervised learning algorithm used for classification based on the majority vote of the nearest neighbors.
- **Decision Tree**: A classification model that makes predictions based on tree-like structures where each node represents a feature decision.

The models were evaluated and compared using accuracy, precision, recall, and F1-score.

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes the necessary libraries, such as:
   - `scikit-learn`
   - `streamlit`
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`

## 🖥️ How to Run

1. To run the Streamlit app, use the following command:
   ```bash
   streamlit run app.py
   ```

   This will open a web interface where you can input patient data and get heart disease predictions.

## 📄 Model Training

1. Train the KNN and Decision Tree models using the provided scripts:
   - `knn.ipynb` for K-Nearest Neighbors
   - `dtree.ipynb` for Decision Tree

2. The trained models are saved and can be reloaded for making predictions.
