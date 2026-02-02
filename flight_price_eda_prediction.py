# %%
# objective of the study is to analyse the flight booking dataset
# conduct hypothesis tests and perform linear regression to predict price of flight

# # Research Questions:
# a) Does price vary with Airlines?
# b) How is the price affected when tickets are bought in just 1 or 2 days before departure?
# c) Does ticket price change based on the departure time and arrival time?
# d) How the price changes with change in Source and Destination?
# e) How does the ticket price vary between Economy and Business class?

# %%
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Load dataset
df = pd.read_csv("Clean_Dataset.csv")

# %%
# Lets' view first 5 rows of data
df.head()

# %%
# View structure of dataset
df.info()

# 12 variables but it really is 11 variables since Unnamed has no significant meaning
# 8 character variables and 3 numerical variables including our predictor which is price


# %%
# Drop unnamed, flight and duration columns since they are not important for this analysis  
df.drop(["Unnamed: 0", "flight", "duration"], axis=1, inplace=True)

# Now we have 9 variables in which 7 are character and 2 are numerical type

# %%
# Let's check to see if there are missing values in the data
df.isnull().sum().sort_values(ascending=False)

# No missing values in dataset

# %%
# Check to see if there are any duplicate observations and if so remove them
df.duplicated().sum()       # No duplicates in data

# %%
# View summary statistics for numerical variables
df.describe()

# %%
# Get the value counts for character variables
df.describe(include=["object"])

# %%
# Let's perform Univariate Analysis for data
# Look at numerical variables first
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# %%
# View histograms
for col in num_cols:
    plt.figure(figsize=(8,6))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()



# %%
# Let's look at boxplots to see if there are any outliers 
for col in num_cols:
    plt.figure(figsize=(8,6))
    sns.boxplot(y=df[col], color="black")
    plt.title(f"Boxplot of {col}")
    plt.show()

# Price has a lot of outliers above the 3rd quartile meaning that their are some expensive flights

# further analysis can be used determine which observations are actually outliers in the data

# %%
# Let's look at bar plots for categorical variables
cat_cols = df.select_dtypes(include=['object']).columns

# %%
for col in cat_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()
    

# %%
# Let's look at correlation heatmaps and scatterplot matrix for numeric variables to see if there is a relationship among them
# Correlation matrix
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

# there is no correlation between price, days_left
# days_left does not have a relationship with flight price

# %%
# Visualize scatterplot matrix
sns.pairplot(df[num_cols])
plt.show()

# %%
# Let's look at first question: Do prices vary with airlines

# %%
# Let's group airlines by price
airline_price = df.groupby('airline')['price'].mean()   # Average airline flight price

# %%
# Visualize
airline_price.plot(kind='bar')
plt.xlabel('Airline')
plt.ylabel('Average Flight Price')
plt.title("Comparison of Flight Price Across Airlines")
plt.legend(title="Airline")
plt.show()

# We can see that Vistara flight price is alot higher than the others. Air India also has higher prices than the other 4
# Seems that prices of airlines differ

# %%
# Let's use ANOVA to test and see if there is significant differences in the average flight prices for each airline
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# %%
# Define the model: Dependent_Variable ~ C(Independent_Variable)
model = ols('price ~ C(airline)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# F-statisic is extremely large indicating that there is a significant different the flight prices between airlines
# At least one airline prices is different than the others

# %%
# Use Tukey's Test to determine which airline prices differ
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_airline = pairwise_tukeyhsd(endog=df['price'], groups=df['airline'], alpha=0.05)
print(tukey_airline)

# From the results below we see that there is a significant difference between most of the airlines except for
# GO_FIRST -> Indigo and GO_FIRST -> SpiceJet

# %%
# Second question
# b) How is the price affected when tickets are bought in just 1 or 2 days before departure?


# %%
sns.scatterplot(
    data=df,
    x="days_left",
    y="price",
    hue="airline"
)

plt.title("Scatterplot of Price vs Days Left by Airline")
plt.show()

# %%
# Does ticket price change based on the departure time and arrival time?


# %%
# Let's groupby departure time and then groupby arrival time
departure_price = df.groupby('departure_time')['price'].mean()
arrival_price = df.groupby('arrival_time')['price'].mean()

# %%
departure_price.plot(kind="bar")
plt.xlabel('Departure Time')
plt.ylabel("Price")
plt.title("Comparing Prices Across Departure Times")
plt.legend(title="depature_time")
plt.show()

# %%
arrival_price.plot(kind="bar")
plt.xlabel("Arrival Time")
plt.ylabel("Price")
plt.title("Comparing Prices Across Arrival Times")
plt.legend(title="arrival_time")
plt.show()

# %%
# Perform ANOVA to see if there is a significant difference in the departure times and arrival times
# Define the model:
model = ols('price ~ C(departure_time)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# There is a significant difference in the prices in the departure times
# The price for either departure or arrival times are different from each other

# Perform Tukey's HSD to see which departure times prices are different from each other
tukey_depart = pairwise_tukeyhsd(endog=df['price'], groups=df['departure_time'], alpha=0.05)
print(tukey_depart)

# %%
# Perform Tukey's HSD to see which departure times prices are different from each other
tukey_depart = pairwise_tukeyhsd(endog=df['price'], groups=df['departure_time'], alpha=0.05)
print(tukey_depart)

# All departure times prices are significantly different from each other

# %%
# Define the model: 
model = ols('price ~ C(arrival_time)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# There is a significant difference in the prices for different arrival times
# Some arrival times prices are either more or less than other arrival times

# %%
# Perform Tukey's HSD to see which arrival times prices are different from each other
tukey_arrival = pairwise_tukeyhsd(endog=df['price'], groups=df['arrival_time'], alpha=0.05)
print(tukey_arrival)

# All arrival times have significantly different prices from one another

# %%
# How the price changes with change in Source and Destination?


# %%
# Groupby source city and destination city by price
source_price = df.groupby('source_city')['price'].mean()
destination_price = df.groupby('destination_city')['price'].mean()

# %%
source_price.plot(kind="bar")
plt.xlabel("Source City")
plt.ylabel("Price")
plt.title("Comparing Prices Across Source City")
plt.legend(title="source_city")
plt.show()

# %%
destination_price.plot(kind="bar")
plt.xlabel("Destination City")
plt.ylabel("Price")
plt.title("Comparing Prices Across Destination Cities")
plt.legend(title="destination_source")
plt.show()

# %%
# Define the model: 
model = ols('price ~ C(source_city)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# There is a significant difference in the flight prices across different source cities

# %%
# Perform Tukey's HSD to see which source cities prices are different from each other
tukey_source = pairwise_tukeyhsd(endog=df['price'], groups=df['source_city'], alpha=0.05)
print(tukey_source)

# %%
# Define the model: 
model = ols('price ~ C(destination_city)', data=df).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# There is a significant difference in flight prices across different destination cities

# %%
# Perform Tukey's HSD to see which destination city prices are different from each other
tukey_desti = pairwise_tukeyhsd(endog=df['price'], groups=df['destination_city'], alpha=0.05)
print(tukey_desti)

# %%
# How does the ticket price vary between Economy and Business class?
# Group class by price
class_price = df.groupby('class')['price'].mean().sort_values()

# %%
class_price.plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Price")
plt.title("Comparing Prices Across Flight Class")
plt.legend(title="class")
plt.show()

# Ticket prices for Business class are a lot more expensive than ticket for economy class

# %%
# Now we can build a model to predict flight price
# Let's encode categorical variables
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# %%
# Select features and target variable
X = df.drop('price', axis=1)
y = df['price']

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
# Fit model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# %%
# Make predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test  = lr_model.predict(X_test)

# %%
# Evalute Model Performance
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nModel Performance:")
print(f"Training MSE: {mse_train:.2f}, R2: {r2_train:.2f}")
print(f"Testing MSE: {mse_test:.2f}, R2: {r2_test:.2f}")

# %%
# Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
})
print("\nLinear Regression Coefficients:\n", coef_df)
print("Intercept:", lr_model.intercept_)

# Extremely large coefficients(maybe scaling can be useful instead of inputing unscaled data)

# %%
# Visualizing
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_test, alpha=0.7, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Linear Regression: Predicted vs Actual")
plt.grid(True)
plt.show()


