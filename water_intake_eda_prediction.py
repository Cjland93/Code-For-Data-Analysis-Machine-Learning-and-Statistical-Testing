# %%
# Analyze  daily water intake and hydration levels of individuals based on a combination of demographic, lifestyle, and environmental factors. 
# What we are going to use/do with the dataset is:
    # Hydration level prediction
    # Lifestyle and wellness analysis
    # Feature importance and correlation studies
    # Binary classification models (Good vs Poor hydration)

# %%
# Import Libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# %%
# Load dataset
df = pd.read_csv("Daily_Water_Intake.csv")

# %%
# View first 5 rows of dataset
df.head()

# %%
# Check structure of dataset
df.info()

# 7 variables: 3 numeric and 4 character
# Character type variables will be recoded later for model

# %%
# view Unique values for each character type variable
for col in df:
    if col == 'Age' or col == 'Daily Water Intake (liters)' or col == 'Weight (kg)':
        continue
    else:
        print(df[col].unique())

# Gender has 2 unique values
# Physical Activity Level has 3 unique values
# Weather has 3 unique values 
# Hydration Level has 2 unique values

# %%
# check for any missing values in data
df.isnull().sum().sort_values(ascending=False)

# There are no missing values in data

# %%
# Drop any duplicates that exist
df.drop_duplicates(inplace=True)

# %%
df.shape

# Dataset had originally 30,000 observations but now there is 29,662 observations
# There were 338 duplicates in the data

# %%
# View summary statistics of numerical values
df.describe()

# %%
# Look at the count for categorical variables
df.describe(include=['object'])

# %%
# Univariate Analysis
# Select numerical columns and categorical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# %%
# View histograms of numerical variables
for col in num_cols:
    plt.figure(figsize=(8,6))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# %%
# Look to see if there are any outliers for numerical variables
for col in num_cols:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col], color='green')
    plt.title(f"Boxplot of {col}")
    plt.show()

# There are outliers for Daily Water Intake possibly indicating that these individuals could be involved in a physically demanding
# activity or it could be possible outliers

# %%
# View countplot of categorical variables
for col in cat_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# %%
# Even amount of females and males
# Even amount of high, moderate and low physical activity levels
# Even amount of hot, normal and col weather
# Majority of individuals have good hydration level


# %%
# Bivariate Analysis

# Look at the correlation among numeric variables
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title('Correlation Matrix')
plt.show()

# Weight and Daiy Water Intake have a strong postive correlation of 0.64
# indicating as Daily Water Intake increases so does weight

# %%
# Let's look at how Age, Daily Water Intake and Weight looks across each Gender by looking at the average of each

gender_summary = df.groupby("Gender").agg({
    "Age": "mean",
    "Daily Water Intake (liters)": "mean",
    "Weight (kg)": "mean"
})

# %%
print(gender_summary)

# The average ages for both male and females are roughly the same, water intke is roughly the same and Females weight slightly more than Males.


# %%
# Let's look at the average age, water intake and weight for each activity level
activity_age = df.groupby("Physical Activity Level")['Age'].mean()
activity_intake = df.groupby("Physical Activity Level")["Daily Water Intake (liters)"].mean()
activity_weight = df.groupby("Physical Activity Level")["Weight (kg)"].mean()

# %%
# Put results into dictionary
activity_dict = {
    "Average Age by Activity Level": activity_age,
    "Average Daily Water Intake by Activity Level": activity_intake,
    "Average Weight by Activity Level": activity_weight
}

# %%
# Loop through dict and plot each
for title, data in activity_dict.items():
    plt.figure(figsize=(8,6))
    sns.barplot(x=data.index, y=data.values, palette="viridis")
    plt.title(title)
    plt.ylabel("Average")
    plt.xlabel("Physical Activity Level")
    plt.show()

# %%
df.info(0)

# %%
# Lastly let's look at Hydration Level and how age, weight and daily water intake compares for each group (good or poor)
hydration_summary = df.groupby("Hydration Level").agg({
    "Age": "mean",
    "Weight (kg)": "mean",
    "Daily Water Intake (liters)": "mean"
})

# %%
# print results
print(hydration_summary)

# Average age for poor hydration is higher than average age for good hydration
# Average weight for poor hydration is higher than average weight for good hydration
# Average water intake for good hydration is higher than average water intake for poor hydration

# Meaning: Individuals with good hydration level tends to be younger, weigh less and consume more water daily than there 
# counterparts who have poor hydration

# %%
# Let's encode our categorical variables 
from sklearn.preprocessing import LabelEncoder

# %%
# Make copy of original dataframe to use for modeling
model_df = df.copy()

# %%
label_enc = {}
for col in cat_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    label_enc[col] = le

# %%
for col, le in label_enc.items():
    print(f"\n--- {col} Mapping ---")
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    for category, integer in mapping.items():
        print(f"{category} -> {integer}")


# %%
model_df.head()

# %%
# Select predictors and target variable
np.random.seed(42)
X = model_df.drop("Hydration Level", axis=1)
y = model_df["Hydration Level"]

# %%
# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# Fit Decision Tree
model_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
model_tree.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model_tree.predict(X_test)

# %%
# Let's evaluate model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# The model is 95% accurate at classifing whether a individual has good or poor hydration level

# %%
# Visualize Tree
plt.figure(figsize=(8,6))
plot_tree(model_tree, feature_names=X.columns,
          class_names=[str(c) for c in np.unique(y)],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Classification")
plt.show()

# %%
# Feature Importance
feat_import = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_tree.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feat_import)

# Weight and Daily Water Intake are the most important features for classifying hydration level
# other predictors can be remove since they have no signifcance/impact on hydration level.


