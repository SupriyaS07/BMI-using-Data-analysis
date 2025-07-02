# p3.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the preprocessed data
print("Loading cleaned dataset...")
df = pd.read_csv('bankmarketing_cleaned.csv')
print(f"Dataset shape: {df.shape}")

# Step 2: Split features and target
print("Splitting features and target...")
X = df.drop('y_yes', axis=1)  # Features
y = df['y_yes']               # Target variable

# Step 3: Train/Test Split
print("Splitting into train and test sets (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Make predictions
print("Making predictions on test data...")
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("\n Model Evaluation Metrics:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_logistic.png")
plt.show()
