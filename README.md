# MNIST Digit Classification 🧠

This project demonstrates various classification techniques on the famous MNIST dataset using **Scikit-Learn**.

### 🔍 Features
- Binary classification (detect digit '5')
- Precision, Recall, F1-score, ROC analysis
- Multiclass classification using **SVM (One-vs-One)**
- Interactive digit testing: choose any digit index to test
- Data scaling using `StandardScaler`

### 📦 Requirements
Install the dependencies:
```bash
pip install -r requirements.txt

You’ll be prompted to enter a digit index (e.g. 1234) from the MNIST test set, and the model will predict it.