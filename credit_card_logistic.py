# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay

# We want to predict whether a transaction is fradulent(1) or legit (0)

# Load dataset
df = pd.read_csv("credit_card_fraud_10k.csv")

# %%
# View first 5 rows of data
df.head()

# %%
# Let's look at the structure of data
df.info()

# There is 10 variables: 9 numerical and 1 character type 

# %%
# Make copy of data
df_2 = df.copy()

# %%
# Remove transaction_id since it is just an identifier
df_2.drop("transaction_id", axis=1, inplace=True)

# %%
# Look at the summary statistics for numerical variables
df_2.describe()

# %%
# View summary statistics for character variables
df_2.describe(include=['object'])

# %%
# Remove any duplicate variables
print(df_2.duplicated().sum())
# No duplicate values

# %%
# Look for any missing values in data
print(df_2.isnull().sum())

# No missing values in dataset

# %%
# Let's view distributions of numerical variables
num_cols = df_2.select_dtypes(include=['int64', 'float64']).columns

# %%
# Histograms
for col in num_cols:
    plt.figure(figsize=(8,6))
    sns.histplot(df_2[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# %%
# Let's look at boxplots of numerical variables
for col in num_cols:
    plt.figure(figsize=(14, 4))
    sns.boxplot(y=df_2[col], color='green')
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()



# %%
# Let's look at barplot for categorical variable(s)
cat_cols = df.select_dtypes(include=['object']).columns

# %%
for col in cat_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df_2, x=col, order=df_2[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# %%
# Let's look to see if there is any correlation among the numerical variables
# 1. Correlation Matrix
corr = df_2[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title("Correlation Matrix")
plt.show()

# Numerical variables are not correlated with each other nor are they correlated with is_fraud

# %%
# Let's perform logistic regression using statsmodels for analyzing model
import statsmodels.formula.api as smf

# Define and fit model
log_model = smf.logit("is_fraud ~ amount + transaction_hour + foreign_transaction + location_mismatch + device_trust_score + velocity_last_24h + cardholder_age", data=df_2).fit()

# %%
# Analyze results
print(log_model.summary())

# All predictors are signficant at determined whether a transaction is fradulent or legit

# %%
# Encode merchant category for model
df_2 = pd.get_dummies(df_2, columns=['merchant_category'], dtype=int,drop_first=True)

# %%
# Let's build logistic regression model to predict whether transaction is fradulent or legit
# Select features and target variable
X = df_2.drop('is_fraud', axis=1)
y = df_2['is_fraud']

# %%
# Split train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# Fit logistic regression
logreg_model = LogisticRegression(max_iter=200, random_state=42)
logreg_model.fit(X_train, y_train)

# %%
# Make Predictions
y_pred = logreg_model.predict(X_test)
y_prob = logreg_model.predict_proba(X_test)[:, 1]

# %%
# Evaluate model
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test accuracy is 99.2% meaning the model is 99.2% accurately at predicting all cases of credit card transaction

# %%
# View confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# True Negative(TN) = 1968 -> correctly predicted legitimate transaction
# False Postive (FP) = 2 -> predicted fradulent but actual outcome was legitimate
# False Negative (FN) = 14 -> predicted legitimate but actual outcome was fraudulent
# True Positive (TP) = 16 -> correctly predicted fradulent transaction

# Accuracy = (TP + TN) / Total = (1968 + 16)/2000 = 99.2% -> model correctly classifies 99.2% of all cases
# Sensitivity (Recall / True Positive Rate) = TP / (TP + FN) = 16/ 30 = 53.3% -> model correctly identifies about 53.3% of Fradulent transactions
# Specificity (True Negative Rate) = TN / (TN + FP) = 14 / 16 = 87.5% -> model correctly identifies 87.5% of legitimate transactions

# %%
# ROC Curve 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2}")
plt.plot([0,1],[0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# AUC = 0.99 -> meaning model has extremely good (almost perfect) discriminative ability
# model can correctly differentiate fradulent and legitimate credit card transactions cases approximately 99% of the time

# %%
# Let's view coefficients and odds ratios of the model
coef_df = pd.DataFrame({'Feature': X.columns,
                        'Coefficient': logreg_model.coef_[0],
                        'Odds_Ratio (OR)': np.exp(logreg_model.coef_[0])})
print("\nLogistic Regression Coefficients:\n", coef_df)
print("Intercept:", logreg_model.intercept_[0])


