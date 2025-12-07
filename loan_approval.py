# %%
# Have a dataset consisting of 2000 individauls in which the goal is to predict whether or not
# they are approved for a loan based on certain features

# Variables include:
# name, city, income, credit score, loan amount, years employed, points, loan_approved

# %%
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

# %%
# Load data
df = pd.read_csv("loan_approval.csv")

# %%
# View first few rows of dataset
df.head()

# %%
# View last 5 rows of dataset
df.tail()

# %%
# Check contents/structure of dataset
df.info()

# Have 5 numeric datatypes, 2 character/object datatypes and 1 boolean datatype

# %%
# Look at the shape of data
df.shape

# Have 2000 observations/ individuals (rows) and 8 variables (columns)

# %%
# Check for missing values
print(df.isnull().sum())

# Have 0 missing values

print()

# Check for any duplicates
print(df.duplicated().sum())

# There are no duplicated values in dataset

# %%
# View summary statistics of numeric values
df.describe().round(0)

# %%
# Income: avg income is $90,586. with income ranging from $30,053.00 to $ 149,964
# Credit Score: avg credit score is 574 with scores ranging from 300 to 850
# Loan Amount: avg loan amount is $25,309 with loans ranging from $1,022 to $49,999
# Years Employed: avg years employed is 20 years with years ranging from 0 to 40 years
# Points: avg points is 57.0 with points ranging from 10 to 100

# %%
# Make copy of dataframe so that old dataset will not be modified
df_new = df.copy()

# %%
# Let's look at the distributions of numeric cols
num_cols = df_new.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df_new[col], kde=True, color="royalblue")
    plt.title(f"Distribution of {col}")
    plt.show()

    # View boxplots of numeric variables
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df_new[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# %%
# Let's look at the counts for loan_approved
print(df_new['loan_approved'].value_counts())

# So we see that out of 2000 individuals, 1,121 (56.1%) of were not approved for loan

# %%
# Let's see if the numeric variables are correlated with one another
corr = df_new[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='icefire')
plt.show()

# We see that credit_score and points are strongly correlated together (0.74)
# Other than that the numeric variables do not have much correlation with each other.

# %%
# Since we want to predict loan approved and being that it is boolean let's convert from 
# boolean to numeric where we want True/Loan Approved = 1 and False/Loan Denied  = 
# Create mapping dictonary
mapping = {True: 1, False: 0}

df_new['loan_approved'] = df_new['loan_approved'].map(mapping)

# %%
# view loan_approved column to see if mapping worked
df_new.head(6)

# %%
# Drop name and city from dataframe
cols_drop = ['name', 'city']
df_new.drop(columns=cols_drop, inplace=True)

# %%
# Perform Logistic Regression
np.random.seed(42)

# Select Features and Target
X = df_new.drop('loan_approved', axis=1)
y = df_new['loan_approved']

# Show the shape of features and target
print("Shape:", X.shape, y.shape)

# %%
# Can do logistic model on entire dataset but we are going to use train and test sets for this
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# We have 1600 training samples and 400 testing samples

# %%
# Standardize Features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Fit Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# %%
# Make Predictions

# Predict class labels
y_pred = model.predict(X_test_scaled)

# Predict probabilities for Loan approved (True)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Display probabilities
print(f"Predicted Probabilites (Loan Approval):\n{y_prob}")

# %%
# Choose classification threshold
threshold = 0.5

# Classify based on threshold
y_pred_threshold = (y_prob >= threshold).astype(int)

# Create Dataframe to hold results/ this is the test set
loan_results = pd.DataFrame({
    'Actual Loan-Approval': y_test,
    'Predicted Loan-Approval': y_pred,
    'Predicted Probabilities': y_prob
})

# %%
# View first 20 observations
loan_results.head(21)

# %%
# Let's put original dataframe labels into Test Set Dataframe for clean view 

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Get categorical columns for test set
X_test_cats = df.loc[X_test.index, cat_cols]

# Ensure X_test is a DataFrame
X_test_num = pd.DataFrame(X_test, columns=X.columns, index=X_test.index)

# Combine everything
loan_results_full = pd.concat(
    [X_test_cats, X_test_num, loan_results],
    axis=1
)

# This corrects display of dataframe columns
pd.set_option('display.max_columns', None)      # show all columns
pd.set_option('display.width', None)            # don't wrap long lines
pd.set_option('display.max_colwidth', None)     # show full content in each cell
pd.set_option('display.expand_frame_repr', False)  # stop printing with '\'


loan_results_full.head()

# %%
# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Classification Report: \n{class_report}")

# Model has 100% Accuracy at predicting True Loan Approval's  and True Loan Denials

# If we used the dataset without standardizing the features these values would be different 
# and accuracy would not be as high

# %%
# Let's look at ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel("False Loan Approval")
plt.ylabel("True Loan Approval")
plt.title("Logistic Regression ROC Curve")
plt.legend(loc='lower right')
plt.grid(True)
plt.show

# Curve is saying that this model is 100%/perfect at distingushing between 
# people who are approve and denied for loan

# %%
## Let's Finally look at the coefficients of the model
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

print("\nLogistic Regression Coefficients:")
print(coef_df)

print("\nIntercept:", model.intercept_[0])

# %%
# Logistic Regression Equation: Loan Approval = -0.413 + 0.12*income + 0.79*credit_score +
# -0.34*loan_amount + 0.088*years_employed + 10.13*points


