from google.colab import files
uploaded = files.upload()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix)

df = pd.read_csv("german_credit_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

df.fillna(df.mode().iloc[0], inplace=True)

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("---------SAMPLE CUSTOMER PREDICTION-----------")

sample = X_test.iloc[0].values.reshape(1, -1)

prediction = model.predict(sample)

print("Actual Value    :", y_test.iloc[0])
print("Predicted Value :", prediction[0])

if prediction[0] == 1:
    print("Customer is Creditworthy")
else:
    print("Customer is Not Creditworthy")

print("---------MODEL PERFORMANCE---------")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
