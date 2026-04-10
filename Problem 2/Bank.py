# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 2. Download and Unzip the dataset
# We use the file ID from your provided link
file_id = '18KwSR9aVTZRNaOVF76VE9USSEkqnYzzQ'
url = f'https://drive.google.com/uc?id={file_id}'

import os
import requests
import zipfile

# Download the file
print("Downloading dataset...")
response = requests.get(url, stream=True)
with open("bank_data.zip", "wb") as f:
    f.write(response.content)

# Unzip the file
print("Unzipping dataset...")
with zipfile.ZipFile("bank_data.zip", "r") as zip_ref:
    zip_ref.extractall("bank_content")

# Locate the CSV file based on your description
# Path: bank_content/bank-data/bank-full.csv
csv_path = 'bank_content/bank-data/bank-full.csv'

# 3. Load the Dataset
# Note: This specific dataset often uses ';' as a separator
df = pd.read_csv(csv_path, sep=';')
print("Dataset Loaded Successfully!")
print(df.head())

# 4. Exploratory Data Analysis (EDA)
print(df.info())
sns.countplot(x='y', data=df)
plt.title('Distribution of Target Variable (Term Deposit Subscription)')
plt.show()

# 5. Data Preprocessing
# Encoding categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Splitting features and target
X = df.drop('y', axis=1)
y = df['y']

# Splitting into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Important for Logistic Regression convergence)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build and Train the Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
