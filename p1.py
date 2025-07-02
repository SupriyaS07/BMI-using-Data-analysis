import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv('bankmarketing.csv')

# First 5 rows
print("First 5 rows:")
print(df.head())

# Check and handle missing values
print("\n Missing Values:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Statistical summary
print("\n🔹 Statistical Summary:")
print(df.describe(include='all'))

# Create a folder to save plots (optional)
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Distribution of numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'plots/dist_{col}.png')
    plt.close()

# # Distribution of categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col != 'y':
        plt.figure(figsize=(8, 3))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, color='skyblue')
        plt.title(f'Countplot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/count_{col}.png')
        plt.close()


# Target variable plot
plt.figure(figsize=(5, 3))
sns.countplot(data=df, x='y', color='skyblue')  
plt.title('Target Variable Distribution')
plt.xlabel('Subscribed (y)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/target_y.png')
plt.show()
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

print("\n✅ All analysis and plots saved to the 'plots/' folder.")
