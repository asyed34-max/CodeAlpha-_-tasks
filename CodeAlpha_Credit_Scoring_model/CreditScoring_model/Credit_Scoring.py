# CREDIT SCORING MODEL

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

#SAMPLE DATASET
data = {
    'Age': [25,35,45,29,50,40,33,55,28,48,30,42,27,39,60,32,37,46,31,52],
    'Income': [40000,60000,30000,50000,25000,70000,45000,20000,52000,28000,
               48000,65000,43000,55000,22000,51000,58000,35000,49000,26000],
    'LoanAmount': [10000,15000,20000,8000,25000,10000,12000,30000,9000,22000,
                   11000,13000,9500,14000,28000,10000,12500,21000,10500,24000],
    'CreditHistory': [1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,0],
    'ExistingDebt': [5000,3000,12000,2000,15000,4000,5000,18000,2500,13000,
                     3500,4500,2800,3800,17000,3000,4200,12500,3200,14500],
    'Creditworthy': [1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,0]
}

df = pd.DataFrame(data)

# TARGET
X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
