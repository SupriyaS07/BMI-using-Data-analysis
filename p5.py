# p5.py

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load preprocessed data
print("Loading cleaned dataset...")
df = pd.read_csv("bankmarketing_cleaned.csv")

# Step 2: Split data into features and target
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Step 3: Split into train/test (to train and save model)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train and save the Random Forest model (best from p4.py)
print("Training Random Forest model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'best_model_rf.joblib')
print("Model saved as 'best_model_rf.joblib'")

# Step 5: Load the model
print("Loading saved model...")
loaded_model = joblib.load('best_model_rf.joblib')

# Step 6: Predict on new customer input
print("Predicting on new data...")

# Example new customer (you can replace with input())
new_data = {
    'age': 45,
    'balance': 800,
    'day': 5,
    'duration': 300,
    'campaign': 1,
    'pdays': -1,
    'previous': 0,
    'job_blue-collar': 0,
    'job_entrepreneur': 0,
    'job_housemaid': 0,
    'job_management': 0,
    'job_retired': 0,
    'job_self-employed': 0,
    'job_services': 1,
    'job_student': 0,
    'job_technician': 0,
    'job_unemployed': 0,
    'job_unknown': 0,
    'marital_married': 1,
    'marital_single': 0,
    'education_secondary': 1,
    'education_tertiary': 0,
    'education_unknown': 0,
    'default_yes': 0,
    'housing_yes': 1,
    'loan_yes': 0,
    'contact_telephone': 0,
    'contact_unknown': 0,
    'month_aug': 0,
    'month_dec': 0,
    'month_jul': 0,
    'month_jun': 1,
    'month_mar': 0,
    'month_may': 0,
    'month_nov': 0,
    'month_oct': 0,
    'month_sep': 0,
    'poutcome_other': 0,
    'poutcome_success': 0,
    'poutcome_unknown': 1
}

# Ensure order of columns is same as training
input_df = pd.DataFrame([new_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Make prediction
prediction = loaded_model.predict(input_df)[0]
prob = loaded_model.predict_proba(input_df)[0][1]

# Display result
print("\nPrediction Result:")
if prediction == 1:
    print("The customer is likely to SUBSCRIBE to the term deposit.")
else:
    print("The customer is NOT likely to subscribe.")

print(f"Confidence: {prob:.2f}")
