# Attributes Information
# 1) CRIM: per capita crime rate by town
# 2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3) INDUS: proportion of non-retail business acres per town
# 4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
# 6) RM: average number of rooms per dwelling
# 7) AGE: proportion of owner-occupied units built prior to 1940
# 8) DIS: weighted distances to five Boston employment centres
# 9) RAD: index of accessibility to radial highways
# 10) TAX: full-value property-tax rate per $10,000 [$/10k]
# 11) PTRATIO: pupil-teacher ratio by town
# 12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13) LSTAT: % lower status of the population

# Target Variable
# 1) MEDV: Median value of owner-occupied homes in $1000's [k$]



# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("boston.csv")

# View first few rows of data
df.head()

# MEDV is our target variable 

# Check the content of dataset
df.info()

# We have 14 variables that are all numerical

# Check for any missing values
df.isnull().sum()       # 0 missing values

# Check for any duplicates
print(df.duplicated().sum())    # There are no duplicate observations

# View summary statistics for each variable
df.describe()

# View histograms for each variable
for col in df:
    plt.figure(figsize=(10,8))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Let's visually look at boxplots to see if there are any outliers present
for col in df:
    plt.figure(figsize=(10,8))
    sns.boxplot(y=df[col], color='black')
    plt.title(f"Boxplot of {col}")
    plt.show()

# There looks to be many outliers present among each variables

# Also look at correlation heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# RM is correlated with MEDV at 0.7, As RM (number of rooms per dwelling),
# increases, MEDV (median value of occupied homes)also increases

# LSAT is correlated negatively strong with MEDV at -0.74 meaning as LSAT 
# (% Lower status of population) increases, MEDV decreases

# Let's Predict MEDV based on the other 13 predictors(features)
np.random.seed(42)

# Define features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Train and test split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shape of the Training and Testing set
print(f"Training sample size: {X_train.shape}")
print(f"Testing sample size: {X_test.shape}")

# Features are on different scales so let's standardize features
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Let's fit Linear regression model
lr_full = LinearRegression()
lr_full.fit(X_train_scale, y_train)

# Let's make predictions
y_pred_train = lr_full.predict(X_train_scale)
y_pred_test = lr_full.predict(X_test_scale)

# Let's Evaluate performance of model on train and test (test is what we are
# focused on more)

mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nModel Performance:")
print(f"Training MSE: {mse_train:.2f}, R2: {r2_train:.2f}")
print(f"Testing MSE: {mse_test:.2f}, R2: {r2_test:.2f}")

# For Training Set R2 is 0.75 which means that 75% of the variation in
# MEDV is explained by the features

# For Testing Set R2 is 0.67 which means that 67% of the variation in MEDV
# is explained by the features. This is okay but can be better

# Let's look at the coefficient for each predictor to see how they contribute to MEDV
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_full.coef_
})

print("\nRegression Coefficients:\n", coef_df)
print("Intercept:", lr_full.intercept_)

# for example RM has coefficient of 3.14. This means that for each additional room
#, the predicted MEDV increases by $3.145 thousand (3,145) on average, holding all
# other variables constant

# Visualize Actual vs Predicted
plt.figure(figsize=(10,8))
plt.scatter(y_test, y_pred_test, alpha=0.7, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Linear Regression: Predicted vs Actual")
plt.grid(True)
plt.show()


