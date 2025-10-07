"""
MNIST Classification Project
Description: Demonstrates various types of classification tasks using the MNIST dataset.
"""
import sys
assert sys.version_info >= (3, 7)
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


# 🧠 Load MNIST Dataset
# =============================
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist["data"], mnist["target"]
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# 🔢 Binary Classification (Is this digit 5?)
# =============================
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print("Prediction for first digit:", sgd_clf.predict([X[0]]))
print("Binary classifier accuracy (3-fold CV):",
      cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# Confusion Matrix and Precision/Recall
# =============================
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print("Confusion matrix:\n", cm)
print("Precision:", precision_score(y_train_5, y_train_pred))
print("Recall:", recall_score(y_train_5, y_train_pred))


# Precision-Recall Tradeoff
# =============================
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

idx_90 = (precisions >= 0.90).argmax()
threshold_90 = thresholds[idx_90]
print("Threshold for 90% precision:", threshold_90)

y_train_pred_90 = (y_scores >= threshold_90)
print("Precision at 90%:", precision_score(y_train_5, y_train_pred_90))
print("Recall at 90% precision:", recall_score(y_train_5, y_train_pred_90))


# ROC Curve
# =============================
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="SGD Classifier")
plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.grid()
plt.legend()
plt.show()

print("ROC AUC Score:", roc_auc_score(y_train_5, y_scores))


# 🌲 Random Forest Comparison
# =============================
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(6, 5))
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall (SGD vs Random Forest)")
plt.legend()
plt.grid()
plt.show()

print("Random Forest F1 Score:", f1_score(y_train_5, y_scores_forest > 0.5))
print("Random Forest ROC AUC:", roc_auc_score(y_train_5, y_scores_forest))


# Multiclass Classification (0–9)
# =============================
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
some_digit = X[0]
print("Predicted label (SVC):", svm_clf.predict([some_digit]))
print("SVC classes:", svm_clf.classes_)

# (Optional) One-vs-Rest Example
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
print("Predicted label (OvR):", ovr_clf.predict([some_digit]))
print("Total estimators (OvR):", len(ovr_clf.estimators_))


# Scaling and SGD Accuracy (Multiclass)
# =============================
sgd_clf = SGDClassifier(random_state=42)
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_small.astype("float64"))

print("Scaled SGD Accuracy (3-fold CV):",
      cross_val_score(sgd_clf, X_train_scaled, y_train_small, cv=3, scoring="accuracy"))

# --- Test the Model on Sample Digits ---
import matplotlib.pyplot as plt

# Ask the user for an index
try:
    index = int(input(f"Enter an index (0 to {len(X_test) - 1}): "))
    if index < 0 or index >= len(X_test):
        raise ValueError("Index out of range.")
except ValueError as e:
    print("Invalid input:", e)
    index = 0  # fallback to first image

# Pick the chosen digit from the test set
some_digit = X_test[index]
some_label = y_test[index]

# Show the image
plt.imshow(some_digit.reshape(28, 28), cmap="gray")
plt.title(f"Actual Label: {some_label}")
plt.axis("off")
plt.show()

# Predict using your trained model (e.g., SVM classifier)
prediction = svm_clf.predict([some_digit])
print(f"Predicted Label: {prediction[0]}")