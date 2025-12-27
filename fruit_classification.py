# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("fruit_classification_dataset.csv")

# Let's view first few rows of data
df.head()

# Check structure of data
df.info()

# We have 3 numeric variables and 4 object variables

# Check shape
print(df.shape)

# 10,000 observations and 7 variables

# Check to see if there are missing values
print(df.isnull().sum())

# No missing values in data

# Check to see if there are any duplicates
print(df.drop_duplicates(inplace=True))

# Dropped 246 values that were duplicates

# Rename price column
df.rename(columns={'avg_price (₹)': "avg_price",
                   'weight (g)': 'weight',
                   'size (cm)': 'size'}, inplace = True)

# View summary statistics for numeric variables
df.describe()

# Average fruit size is 8.61 cm. Sizes range from 0.9 cm to 27.5 cm
# Average weight of fruit is 466.8 g. Weights range from 4.5 g to 3299.8 g
# Average price in rupees is 78.42. Prices range from 9 rupees to 165 rupees

# Look at the number of unique fruits
df['fruit_name'].nunique()

# There are 20 different fruits

# View unique fruit names
df['fruit_name'].unique()

# Let's look at counts for object variables
for col in df.select_dtypes(include=['object']).columns:
    print(df[col].value_counts())

# Majority of the fruit shapes are either round (4796) or oval (4456)
# 74.4 % of all fruits (7261) taste sweet

# view distributions of numeric variables
num_cols = df.select_dtypes(include=['float64']).columns

# Set up subplots
fig, ax = plt.subplots(2, 2, figsize=(8, 8))

for col, axis in zip(num_cols, ax.ravel()):
    sns.histplot(df[col], kde=True, ax=axis)
    axis.set_title(f"Distribution of {col}")

# Hide the unused 4th subplot
ax[1,1].set_visible(False)

plt.tight_layout()
plt.show()

# Distributions of the numeric variables

# Let's look at boxplots for numeric variables
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# We see that size and weight have a lot of outliers

# Let's view bar charts / Count plots for object variables
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, palette="viridis")
    plt.title(f"Countplot of {col}")
    plt.ylabel("Count")
    plt.xlabel(f"{col}")
    plt.xticks(rotation=45)
    plt.show()

# Let's look at the average price per fruit
fruit_avg = df.groupby("fruit_name")["avg_price"].mean()

fruit_avg.plot(kind="bar", figsize=(7,5))
plt.title("Average Price of Fruit")
plt.ylabel("Price")
plt.show()

# we see that dragonfruit and watermelon have the highest 
# average prices

# Grapes have the lowest average price

# Let's also look at the average weight of fruit
weight_avg = df.groupby("fruit_name")["weight"].mean()

weight_avg.plot(kind="bar", figsize=(7,5))
plt.title("Average Weight of Fruit")
plt.ylabel("Weight")
plt.show()

# Watermelons on average have the highest weight
# blueberry, cherry, and grape have the lowest weight on average

# Let's look to see if there is any correlation among the numerical variables
sns.pairplot(df[num_cols])
plt.show()

# Looking at the scatterplot matrix, there seems to be an association among
# the numerical variables

# Let's look at the correlation matrix
corr = df[num_cols].corr()
print(corr)

# We see that size and weight have a 0.92 correlation (very strong positive correlation)
# as size increase so do weight

# We see the average price has a moderately positive correlation
# between size (0.588) and weight (0.546)

# Create copy of dataframe for KNN 
df_new = df.copy()

# Now let's Perform KNN to predict fruit name
# first we need to encode color, shape, taste and fruit name

# Let's perform one-hot encoding to color shape and taste
df_new = pd.get_dummies(df_new, columns=['color', 'shape', 'taste'], drop_first=True, dtype=int)

# Perform LabelEncoder on fruit_name (target_variable)
le = LabelEncoder()

# Fit and Transform Fruit name
df_new['fruit_name'] = le.fit_transform(df_new['fruit_name'])

# We can also get mapping to of original labels to encoded values
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)

df_new.info()

# Set seed for reproducibility
np.random.seed(42)

# Select Features and Target
X = df_new.drop(columns=['fruit_name'])
y = df_new['fruit_name']
model = KNeighborsClassifier(n_neighbors=5)

print("Shape:", X.shape, y.shape)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 7,803 are used for training the model
# 1,951 are used for testing 

# Standardize features: Required
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit KNN Model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy = 1.00 which means that the model is 100% accurate
# at classifying fruit names

# Test multiple k values to find best performance
k_vals = range(1, 21)
accuracy_list = []

for k in k_vals:
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train_scaled, y_train)
    y_pred_temp = temp_model.predict(X_test_scaled)
    accuracy_list.append(accuracy_score(y_test, y_pred_temp))

plt.figure(figsize=(7,5))
plt.plot(k_vals, accuracy_list, marker='o', color="green")
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs k')
plt.grid(True)
plt.show()

# The graph is say that from k = 1 to k = 20 there is 100% accuracy
# for each k 




