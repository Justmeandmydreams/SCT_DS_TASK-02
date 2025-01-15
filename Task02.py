# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C://Users//DELL//Downloads//train.csv"  # Replace with your path if needed
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Cleaning
# Fill missing 'Age' with the median
data['Age'] = data['Age'].fillna(data['Age'].median())

# Fill missing 'Embarked' with the most common port
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop 'Cabin' column due to high number of missing values
data = data.drop(columns=['Cabin'])

# Drop rows with missing values in the target column (if any)
data = data.dropna(subset=['Survived'])

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Drop non-numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])

# Exploratory Data Analysis (EDA)
# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Distribution of target variable
sns.countplot(x='Survived', data=data, palette='Set2', hue='Survived', dodge=False, legend=False)
plt.title('Survival Distribution')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
corr_matrix = numeric_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Analyze survival rate by gender
sns.countplot(x='Survived', hue='Sex', data=data, palette='Set1')
plt.title('Survival by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Analyze survival rate by passenger class
sns.countplot(x='Survived', hue='Pclass', data=data, palette='Set3')
plt.title('Survival by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Age distribution and survival
plt.figure(figsize=(10, 6))
sns.histplot(data, x='Age', hue='Survived', bins=30, kde=True, palette='viridis')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Analyze survival rate by embarkation port
sns.countplot(x='Survived', hue='Embarked', data=data, palette='coolwarm')
plt.title('Survival by Embarkation Port')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Check interaction between Fare and Survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=data, hue='Survived', dodge=False, legend=False)
plt.title('Fare by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Fare')
plt.show()

# Summary of findings
print("\nObservations:")
print("1. Survival rate is higher for females compared to males.")
print("2. Higher passenger classes (Pclass 1) show better survival rates.")
print("3. Age shows distinct patterns, with children having a better survival rate.")
print("4. Higher fares are associated with higher survival chances.")
print("5. Most passengers embarked from port 'S', and survival rates vary by embarkation point.")

