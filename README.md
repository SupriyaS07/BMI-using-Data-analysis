# BMI-using-Data-analysis

Project: Bank Marketing Subscription Prediction
📄 Overview
Developed a machine learning pipeline to predict whether a customer will subscribe to a term deposit based on personal and campaign-related data.

Built using Python and trained multiple classification models.

Includes data preprocessing, visualization, model evaluation, and prediction on new data.

Technologies Used
Python 3

Pandas, NumPy – Data handling & preprocessing

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning models

Joblib – Model serialization

Jupyter Notebook / VS Code – Development environment

📊 Dataset
Source: UCI Machine Learning Repository

Rows: 4521 customers

Target Variable: y (Subscribed: Yes/No)

Features: Age, job, marital status, contact method, campaign duration, etc.

🔍 Project Structure
p1.py – Exploratory Data Analysis (EDA) + plots

p2.py – Data preprocessing and feature encoding

p3.py – Logistic Regression model + evaluation

p4.py – Model comparison (Decision Tree, Random Forest, Logistic) + ROC curve

p5.py – Save/load best model and predict on new input

plots/ – Folder containing all saved graphs and visuals

bankmarketing_cleaned.csv – Final preprocessed dataset

best_model_rf.joblib – Serialized model for deployment

✅ Features
Full EDA and visual insights

Preprocessed and encoded all categorical variables

Compared multiple classification models

Evaluated models using accuracy, confusion matrix, and AUC

Saved the best model and made predictions on new customer data

🔚 Outcome
Best Model: Random Forest Classifier (Accuracy ~90%, AUC ~0.89)

Use Case: Helps banks identify potential customers likely to subscribe to term deposits.

