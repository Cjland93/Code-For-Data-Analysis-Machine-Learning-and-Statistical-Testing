# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# We have 5 medications: drug A, drug B, drug C, drug X and drug Y. We want to predict which drug may be appropriate for future patients
# based off of Age, Sex, Blood Pressure, Sodium to Potassium and Cholesterol of patients.

# Load data 
df = pd.read_csv("drug200.csv")

# View first few rows of data
df.head()

# Check structure of data
df.info()

# 200 observations with 6 variables (5 predictors and 1 target (Drug))

# Check for missing values 
df.isnull().sum()

# Their are no missing values in data

# Check for any duplicates
print(df.duplicated().sum())    # Data does not have any duplicates

# Check unique values for Sex, Cholesterol and Drug
cols = ["Sex", "Cholesterol", "Drug"]

for col in cols:
    print(f"Unique values for {col}")
    print(df[col].unique())
    print()

# View summary statistics for all variables
df.describe(include=['object', 'int64', 'float64'])

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Look at distribution of Numerical variables and Boxplots

for col in num_cols:
    plt.figure(figsize=(8,6))
    
    # Histograms
    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)

    # Boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[col], color='blue')
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)

    plt.show()

# Sodium to Potassium is postively skewed 
# There are no outliers present 

cat_cols = df.select_dtypes(include=['object']).columns

# Let's view Count plots for character variables
for col in cat_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Let's prepare to perform Decision Tree Classification
# Need to convert character variables/columns to numerical to use for Decision Trees

from sklearn.preprocessing import LabelEncoder

char_cols = ["Sex", "BP", "Cholesterol", "Drug"]

# Create dictionary that stores encoders
encoders = {}

# Fit and transform each column
for col in char_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# View dataframe to see if encoding worked
df.head()

# drugA = 0, drugB = 1, drugC = 2, drugX = 3, drugY = 4

# Set random seed for reproducibility
np.random.seed(42)

# Select features and target variable
X = df.drop('Drug', axis=1)
y = df["Drug"]

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Check shape of training and testing sets
print(f"X Train Size: {X_train.shape}")
print(f"X Test Size: {X_test.shape}")
print(f"Y Train Size: {y_train.shape}")
print(f"Y Test Size: {y_test.shape}")

# Fit decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\nClassification Report:\n", classification_report(y_test, y_pred))

# Model a 98% Accuracy meaning that the model can correctly predict the best drug for a future patient 98% of the time.

# Visualize Tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns,
          class_names=[str(c) for c in np.unique(y)],
          filled=True, rounded=True, fontsize=10)
plt.title(f"Decision Tree")
plt.show()

# Let's look at the importance of each feature
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feat_importance)

# Sodium to Potassium is the most important feature in classifying correct drug, followed by BP (Blood Pressure)
# Sex does not have any importance in the model and can actually be removed from the model. 


