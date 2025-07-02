# p4.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Step 1: Load the preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('bankmarketing_cleaned.csv')

X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Step 2: Train/Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Step 4: Train, Predict, and Evaluate Each Model
model_results = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Needed for ROC curve

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    model_results[name] = {
        "model": model,
        "accuracy": acc,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 5: Plot ROC Curve for all models
plt.figure(figsize=(8, 6))
for name, result in model_results.items():
    fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_comparison.png")
plt.show()

# Step 6: Summary
print("\n🔚 Model Comparison Summary:")
for name, result in model_results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.4f}, AUC = {result['auc']:.4f}")
