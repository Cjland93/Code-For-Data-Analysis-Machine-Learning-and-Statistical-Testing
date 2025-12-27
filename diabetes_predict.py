# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# View first few rows of data
df.head()

# variables include: Gender, Age, Hypertension, Heart Disease, Smoking History,
# BMI, HBA1C Level, Blood Glucose Level, and Diabetes (Target)

# Check the structure of the data
df.info()

# There are 100,000 observations with 9 columns(variables)
# gender and smoking history are the only categorical variables

# Check to see if there any missing values
print(df.isnull().sum())
# No missing values in Data

# Check to see if there are any duplicates
print(df.duplicated().sum())    # There are 3,854 duplicates
df.drop_duplicates(inplace=True)    # Drop duplicates from dataset

# Get summary statistics for numerical variables
df.describe()

# Age: average age is approximately 42, ages range between 0 - 80
# BMI: average BMI is 27.3 with values ranging from 10 - 95.69
# HbA1c_Level: average hba1c is 5.53 with values ranging from 3.5 - 9
# Blood glucose Level: average is 138.2 with levels ranging from 80 - 300


# Let's look at the counts for gender and smoking history
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x=col)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.show()
    print(df[col].value_counts()) # Shows the values for each

# Females make up 58.4 % of the Genders
# Never smoked and No information are the highests values for smoking history


# Let's view the distribution of the numeric variables
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Let's look at boxplots to see if there are any outliers in the variables
for col in num_cols:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# We can see that there are outliers for hypertension, heart disease,
# bmi, hba1c level, blood glucose level and diabetes

# we can employ mahalanobis distance to see which individuals in particular
# are outliers

# Since we are just focusing on predicting diabetes we will retain
# the outliers for further analysis

# Let's view scatterplot matrix of numeric variables
sns.pairplot(df[num_cols])
plt.show()

# Let's view heatmap 
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# From the heatmap we see that blood glucose level and hba1c level
# are the highest predictors of diabetes wih 0.42 and 0.41 correlation
# respective. These values are not the strong however. 

# Make a copy of original dataframe to use for modeling
df_model = df.copy()

# Now we can are ready to Predict whether a person has diabetes 
# based on the features
np.random.seed(42)

# Define numeric and categorical columns
categorical = ["gender", "smoking_history"]
numerical = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]

X = df_model[categorical + numerical]
y = df_model["diabetes"]

# Split data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Size for training and testing sets
training = X_train.shape[0]
testing = X_test.shape[0]

print(f"Training samples: {training}")
print(f"Testing samples: {testing}")

# Let's One Hot Encode out categorical columns for the models
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat = encoder.fit_transform(X_train[categorical])
X_test_cat = encoder.transform(X_test[categorical])

X_train_cat = pd.DataFrame(
    X_train_cat, 
    columns=encoder.get_feature_names_out(categorical),
    index=X_train.index
)

X_test_cat = pd.DataFrame(
    X_test_cat, 
    columns=encoder.get_feature_names_out(categorical),
    index=X_test.index
)

# Keep numerical features
X_train_num = X_train[numerical].reset_index(drop=True)
X_test_num = X_test[numerical].reset_index(drop=True)

# Scale for Logistic Regression only
scaler = StandardScaler()

X_train_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_num),
    columns=numerical
)

X_test_num_scaled = pd.DataFrame(
    scaler.transform(X_test_num),
    columns=numerical
)

# Combine numerical and categorical
# Logistic Regression
X_train_log = pd.concat([X_train_num_scaled, X_train_cat.reset_index(drop=True)], axis=1)
X_test_log = pd.concat([X_test_num_scaled, X_test_cat.reset_index(drop=True)], axis=1)

# Decision Trees
X_train_tree = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
X_test_tree = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

# Train Both Models
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_log, y_train)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train_tree, y_train)

# Make Predictions
log_pred = log_reg.predict(X_test_log)
tree_pred = tree.predict(X_test_tree)

from sklearn.metrics import ConfusionMatrixDisplay

# Logistic Regression Evaluation
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

# Visual confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    log_pred,
    cmap="Blues",
    colorbar=True
)
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# Decision Tree Evaluation
print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print(confusion_matrix(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

# Visual confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    tree_pred,
    cmap="Greens",
    colorbar=True
)
plt.title("Confusion Matrix: Decision Tree")
plt.show()

# Classification Results for Logistic Regression:
# True negatives (TN): 17371 -> correctly predicted no diabetes
# False postive (FP): 163 -> predicted diabetes but actual was no diabetes
# False negative (FN): 613 -> predicted no diabetes but actual was diabetes
# True positive (TP): 1083 -> Correctly predicted diabetes

# Accuracy: (TP + TN) / Total = 95.96%
# Logistic model correctly classfied approximately 96% of observations
# logistic model performs exceptionally well

## Classification Results for Decision Tree:
# Accuracy: 97.16 % => decision tree performs exceptionally well,
# a little better than logistic model

# Let's look at ROC Curves for each

# Predicted probabilities
log_probs = log_reg.predict_proba(X_test_log)[:, 1]
tree_probs = tree.predict_proba(X_test_tree)[:, 1]

# ROC values
fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_probs)

# AUC scores
auc_log = auc(fpr_log, tpr_log)
auc_tree = auc(fpr_tree, tpr_tree)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {auc_log:.3f})")
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {auc_tree:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.show()

# AUC for both are relatively high with Logistic Regression being: 0.960
# and Decision Tree being 0.949

# This means that for both models, if one person is selected randomly with diabetes
# and the other without diabetes, the model will correctly assign a higher predicted risk
# to diabetic persons about 96%(Logistic), 95% for Decision Tree

# Both of these model have excellent predictive accuracy with Logistic Regression being slighly better
# than decision tree.

# Let's Look at the Feature Importance for each

import pandas as pd
import matplotlib.pyplot as plt

# Logistic Regression coefficients
log_importance = pd.Series(
    log_reg.coef_[0],
    index=X_train_log.columns
).sort_values(key=abs, ascending=False)

print("\nTop Logistic Regression Coefficients:")
print(log_importance.head(10))

# Decision Tree feature importances
tree_importance = pd.Series(
    tree.feature_importances_,
    index=X_train_tree.columns
).sort_values(ascending=False)

print("\nTop Decision Tree Feature Importances:")
print(tree_importance.head(10))

# Plots
plt.figure(figsize=(10, 6))
log_importance.head(10).sort_values().plot(kind="barh", color="steelblue")
plt.title("Top Logistic Regression Coefficients")
plt.show()

plt.figure(figsize=(10, 6))
tree_importance.head(10).sort_values().plot(kind="barh", color="darkgreen")
plt.title("Top Decision Tree Feature Importances")
plt.show()


# From the Logisic Regression Model:
# HbA1c Level coeefficient: 2.48
# Blood glucose level coefficient: 1.36

# For Decision Tree:
# HbA1C level is the strongest predictor of Diabetes at 0.653
# this means that a person with a higher hba1c level will more likely
# have diabetes versus a person who has a low hba1c level
# blood glucose level importance is 0.330
# the two combine is 0.987, these two account for almost 100%
# of feature importance. this suggests that the other features
# can be removed since they have very little importance. 

# We see that both models work exceptionally well for predicting
# diabetes outcome of individuals. 

# HbA1c Level and Blood Glucose Level are the most important features
# when predicting if a individual has diabetes or not.



