# 🧠 Mental Health Risk Prediction using Apache Spark

This project focuses on predicting the likelihood of mental health issues using survey-based data and machine learning techniques. It leverages **Apache Spark (PySpark)** for distributed data processing and scalable model training.

---

## 📌 Project Overview

Mental health issues such as stress, anxiety, and depression are increasing in modern workplaces. Early prediction can help in timely intervention and support.

This project uses a machine learning approach to classify individuals based on their mental health risk using survey data.

---

## ⚙️ Technologies Used

- Apache Spark (PySpark)
- Python
- Spark MLlib
- Pandas
- Seaborn & Matplotlib

---

## 📊 Dataset

- Source: OSMI Mental Health Survey (Kaggle)
- Type: Survey-based dataset
- Features Used:
  - Age
  - Gender
  - Family History
  - Remote Work
  - Company Size

- Target Variable:
  - Mental Health Risk (Yes/No)

---

## 🔄 Workflow
Dataset → Data Cleaning → Feature Engineering → Model Training → Prediction → Evaluation → Visualization

---

## 🧹 Data Preprocessing

- Removed unnecessary columns
- Handled missing values
- Normalized categorical data (e.g., gender)
- Selected relevant features

---

## 🧠 Feature Engineering

- StringIndexer for categorical encoding
- OneHotEncoder for vector representation
- VectorAssembler for feature combination

---

## 🤖 Model Used

- Logistic Regression (Spark MLlib)
- Binary classification problem
- Train-test split: 80:20

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- F1 Score

Example Results:
Accuracy: ~0.70
F1 Score: ~0.70

(Note: Results may vary slightly due to randomness in data splitting)

---

## 📊 Visualization

- Count plots used to analyze:
  - Mental health risk distribution
  - Risk vs Gender

---

## 📸 Sample Outputs

(Add your screenshots here)
- Dataset preview
- Feature vector
- Predictions
- Accuracy output
- Graph

---
