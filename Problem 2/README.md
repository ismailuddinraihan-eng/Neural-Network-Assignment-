# 🏦 Bank Marketing Prediction using Logistic Regression

## 📌 Overview

This project aims to predict whether a bank customer will subscribe to a term deposit based on their demographic and financial information. The model is built using **Logistic Regression**, a widely used classification algorithm.

The dataset used is the **Bank Marketing Dataset**, which contains customer-related information such as age, job, marital status, balance, and previous interactions with the bank.

---

## 🎯 Objective

To develop a machine learning model that can:

* Predict customer subscription (Yes/No)
* Assist banks in targeted marketing strategies

---

## 📂 Dataset Description

* Source: Bank Marketing Dataset
* Total Features: 17
* Target Variable: `y` (Yes = subscribed, No = not subscribed)

### Key Features:

* Age
* Job
* Marital Status
* Education
* Balance
* Housing Loan
* Contact Type
* Campaign Information

---

## ⚙️ Methodology

### 🔹 1. Data Collection & Loading

* Dataset downloaded from Google Drive
* Extracted using `zipfile`
* Loaded using **Pandas**

---

### 🔹 2. Exploratory Data Analysis (EDA)

* Checked dataset structure using `.info()`
* Visualized target variable distribution using count plot
* Observed class imbalance in the dataset

---

### 🔹 3. Data Preprocessing

* Categorical variables encoded using **LabelEncoder**
* Features (`X`) and target (`y`) separated
* Data split into training and testing sets (80/20)
* Feature scaling applied using **StandardScaler**

---

### 🔹 4. Model Building

* Algorithm: **Logistic Regression**
* Max iterations set to 1000 for better convergence
* Model trained using training data

---

### 🔹 5. Model Evaluation

* Accuracy Score used for performance measurement
* Confusion Matrix used for detailed classification analysis
* Classification Report includes:

  * Precision
  * Recall
  * F1-score

---

## 📊 Results

* ✅ Accuracy: ~80% (approx)
* Model performs well in predicting majority class
* Some imbalance observed in prediction of minority class

---

## 🔍 Key Findings

* Logistic Regression is effective for binary classification problems
* Feature scaling improves model performance
* Dataset is slightly imbalanced, affecting prediction quality
* Model can be improved with advanced techniques

---

## 🚀 Future Improvements

* Handle class imbalance using:

  * SMOTE
  * Class weights
* Try advanced models:

  * Random Forest
  * XGBoost
* Perform hyperparameter tuning
* Feature selection for better performance

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📌 Conclusion

This project demonstrates how Logistic Regression can be applied to predict customer behavior in banking. The model provides a solid baseline and can be further enhanced using advanced machine learning techniques.

---

