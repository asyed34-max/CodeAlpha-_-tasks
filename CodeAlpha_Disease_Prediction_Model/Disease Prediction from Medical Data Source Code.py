from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix)

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("---SINGLE SAMPLE---")

sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

print("Actual Value:", y_test[0])

if prediction[0] == 0:
    print("Predicted Result: Malignant Tumor")
else:
    print("Predicted Result: Benign Tumor")

print("-----MULTIPLE SAMPLES-----")

for i in range(5):

    sample = X_test[i].reshape(1, -1)

    pred = model.predict(sample)[0]

    actual = y_test[i]

    pred_label = "Benign" if pred == 1 else "Malignant"
    actual_label = "Benign" if actual == 1 else "Malignant"

    print(f"\nPatient {i+1}")
    print("Actual   :", actual_label)
    print("Predicted:", pred_label)

print("-----MODEL PERFORMANCE-----")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
