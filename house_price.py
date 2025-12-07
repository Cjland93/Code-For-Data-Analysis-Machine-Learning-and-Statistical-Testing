# Dataset consists of 13 variables (12 features) used to predict house price


# %%
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# %%
# Load data
df = pd.read_csv("Housing.csv")

# %%
# View first few rows of data
df.head()

# Price is what we want to predict ultimately

# %%
# print the shape of dataset
print(df.shape)

# Data has 545 observations with 13 columns/variables

# %%
# Check the contents of data
df.info()

# 6 numeric and 7 character type variables

# %%
# Check for missing values
print(df.isnull().sum())

# No missing values in data

# Check for any duplicates
dups = df.duplicated().sum()
print(f"\nData has {dups} duplicates")

# Data has 0 duplicate values

# %%
# Set display option 
pd.set_option('display.float_format', '{:.2f}'.format) 

# Let's view summary statistics for numerical variables
df.describe().round()

# Price: Avg price is $4,766,729 with prices ranging from $1,750,000
# to $ 13,300,000

# Area: The total area of the house in square feet.
# Avg area is 5,151 sq ft with area ranging from 1,650 to 16,200 sqft

# Bedrooms: Avg number of bedrooms is 3 with bedrooms ranging from 1 to 6
# Bathrooms: Avg number of bathrooms is 1 with bathrooms ranging from 1 to 4
# Stories: Avg number of stories in house is 2 and ranging from 1 to 4 stories
# Parking: Avg number of parking spaces is 1 and spaces range from 0 to 3


# %%
# Let's look at the counts for categorical variables
for col in df.select_dtypes(include=['object']).columns:
    print(df[col].value_counts())

# Mainroad: Whether the house is connected to the main road
# 468 (85.9%) are connected to mainroad while 77 are not

# Guestroom: 448 out of 545 houses (82.2%) have a guest room

# Basement: 

# %%
# Let's perform Univariate Analysis of numeric and character variables
nums_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Look at Histograms/Distribution of Numeric Variables
for col in nums_cols:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(df[col], fill=True, linewidth=2, color='blue')
    plt.title(f"Distribution of {col.capitalize()}", fontsize=13, fontweight='bold')
    plt.xlabel(col.capitalize())
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

# Distribution of Price is skewed heavily to the right (Positively skewed)
# Distribution of Area is also positively skewed

# %%
# Let's look at the boxplots for the numeric variables to see if there
# are any possible outliers
for col in nums_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# All the variables have outliers

# %%
# Let's view barplots of categorical variables

cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, palette="plasma")
    plt.title(f"Countplot of {col}")
    plt.ylabel("Count")
    plt.xlabel(f"{col}")
    plt.xticks(rotation=45)
    plt.show()

# From the barplots we see that:
# Most houses have a mainroad, no guestroom, no basement, no hot water heater,
# no air conditioning and no prefarea

# Most houses are semi-furnished followed by not-furnished and then,
# completely furnished

# %%
# Let's see if there are correlation among numerical variables
sns.pairplot(df[nums_cols])
plt.show()

# %%
# Also use a heatmap
corr = df[nums_cols].corr()
sns.heatmap(corr, annot=True, cmap='rocket')
plt.show()

# From heatmap we see that area is correlated with price (0.54),
# followed by bathrooms being correlated with price (0.52)

# These a moderately positive correlations and suggest that area and
# bathrooms tend to influence the price of houses.

# %%
# Let's Use Random Forest Regression to Predict House Prices

# %%
# Create a copy of original dataframe to use for modeling
df_model = df.copy()

# %%
# Let's first encode the object variables to numerical variables
# Use one hot encoding (since object features have no order)
cat_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
            'prefarea', 'furnishingstatus']
df_model = pd.get_dummies(df, columns=cat_features, drop_first=True)


# %%
df_model.info()

# %%
np.random.seed(42)

# Choose predictors and target
X = df_model.drop("price", axis=1, inplace=False)
y = df_model['price']
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Shape:", X.shape, y.shape)

# %%
# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Training has 436 and Testing has 109

# %%
# Tree based model so scaling is not required


# %%
# Fit Random Forest Model
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)

# %%
# Let's create Dataframe of Test Set to view predictions vs actual
test_df = X_test.copy()
test_df['Actual House Price'] = y_test
test_df['Predicted House Price'] = y_pred

# This corrects display of dataframe columns (For assurance)
pd.set_option('display.max_columns', None)      # show all columns
pd.set_option('display.width', None)            # don't wrap long lines
pd.set_option('display.max_colwidth', None)     # show full content in each cell
pd.set_option('display.expand_frame_repr', False) 

print("Testing Data with Predictions:")
test_df.head()

# %%
# Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nRegression MSE: {mse ** 0.5:.2f}, R2: {r2:.2f}")

# R2 of 0.61 suggesting that the features explain 61% of the 
# total variance/variation of House Price

# Mean squared error says that the predictions vs reality
# of house prices are off about 1,400,565.97 dollars

# %%
# Let's to see which features are the most important
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(feat_importance)

# We see that area is the overwhelming contributor to the price of Houses at 0.47
# followed by bathrooms at 0.15
# Majority of the features have very low importance suggesting that
# they may not contribute to the house price

# %%
# Visualize Feauture Importance
plt.figure(figsize=(8,5))
plt.bar(feat_importance['Feature'], feat_importance['Importance'], color="royalblue")
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.grid(True)
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()                   
plt.show()

# We can visual see how the importance of each feature is for predicting 
# house prices using random forest regression


