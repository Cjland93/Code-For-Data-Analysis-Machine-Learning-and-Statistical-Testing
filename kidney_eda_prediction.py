
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import xgboost as xgb


# Load data 
df = pd.read_csv("kidney_dataset.csv")


# View first 5 rows of data
df.head()

# Our target variable is GFR ( Glomerular Filtration Rate (5–120)) for kidney
# higher => better kidney function

# Let's view contents of data
df.info()

# All variables are numerical except for medication
# Dataset consists of 5,000 observations


# Check to see if there are missing values in data
df.isnull().sum()
# We see that we have 2,987 missing values for medication (~ 59% of medication is null)

# Let's see how many unique values exist for medication
df["Medication"].unique()

# ACE Inhibitor, Diurctice and ARB are unique values with a bunch of NAN

# Let's create new dataframe from original
df_2 = df.copy()


# Drop Medication Column
df_2.drop("Medication", axis=1, inplace=True)

# Check to see if there are any duplicate observations if so drop them
print(df_2.duplicated().sum())
# There are no duplicate observations (rows)

# View summary statistics of dataframe
df_2.describe()

# Check to see if there are any outliers for particular variables
# Let's access boxplots
for col in df_2.select_dtypes(include=["int64", 'float64']).columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(x=df_2[col])
    plt.title(f"Distribution of {col}")
    plt.show()


# There are seems to be a lot of outliers for variables

# Let's perform some feature pattern analysis
# Look at the correlation heatmap 
plt.figure(figsize=(10,8))
sns.heatmap(df_2.corr(), annot=True, cmap = "coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# We see that Urine Output has a strong correlation with GFR of 0.83
# This suggests that high daily urine volume, the higher the GFR

# Let's view a Scatterplot matrix 
sns.pairplot(df_2, diag_kind="kde")
plt.show()

# Check for outliers using Mahalanobis Distance
# Create a dataframe from numeric for outlier analysis
df_md = df_2.copy()

from scipy.stats import chi2
# Compute distances of each point from mean
mean_vec = df_md.mean().values
cov_matrix = np.cov(df_md.values, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

mahal_distances = df_md.apply(
    lambda row: distance.mahalanobis(row.values, mean_vec, inv_cov_matrix),
    axis=1
)

df_md['Mahalanobis'] = mahal_distances

# Correct threshold (square root of chi-square cutoff)
threshold = np.sqrt(chi2.ppf(0.975, df=df_md.shape[1]))

df_md['Outlier'] = df_md['Mahalanobis'] > threshold

num_outliers = df_md['Outlier'].sum()
print(f"\nNumber of outliers detected: {num_outliers}")

# Visualize Outliers
plt.figure(figsize=(10,6))
plt.scatter(range(len(df_md)), df_md['Mahalanobis'], 
            c=df_md['Outlier'].map({True: 'red', False: 'blue'}),
            alpha=0.7)

plt.axhline(threshold, color='black', linestyle='--', label='Threshold')

plt.xlabel("Observation Index")
plt.ylabel("Mahalanobis Distance")
plt.title("Outlier Detection Using Mahalanobis Distance")
plt.legend()
plt.show()

# We can see that their are alot of outliers in the data

# Let's Perform PCA to for dimension reduction

# Define Features
features = ["Creatinine", "BUN", "Urine_Output", "Diabetes", "Hypertension",
"Age", "Protein_in_Urine", "Water_Intake", "CKD_Status"]

X = df_2[features].values
y = df_2['GFR']

# Standardize Data
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

# Let's check to see if 3 Principal Components explained majority of dataset variation
pca = PCA(n_components=3)
X_pca_full = pca.fit_transform(X_Scaled)

print("\nExplained Variance by Components:", pca.explained_variance_ratio_)
# We see that 3 Principal Components Account for about 70% of the total variation
# PC1: 47.1
# PC2: 11.7
# PC3: 11.2

# Create labels for x-axis
pc_labels = np.arange(1, len(pca.explained_variance_ratio_) + 1)

plt.figure(figsize=(10,6))

# Plot individual variance
plt.bar(pc_labels, pca.explained_variance_ratio_, color='skyblue', label='Individual Explained Variance')
plt.plot(pc_labels, pca.explained_variance_ratio_, marker='o', linestyle='-', color='darkred', markersize=8)

# Plot cumlative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(pc_labels, cumulative_variance, marker='s', linestyle='--', color='navy', markersize=8, label='Cumulative Explained Variance' )

plt.xlabel('Principal Component')
plt.ylabel('Proportion of Explained Variance')
plt.title('Scree Plot and Cumulative Variance')
plt.xticks(pc_labels)
plt.legend()
plt.grid(True)
plt.show()

# Lets view the loadings of each Principal Component
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=features
)
print("\nFeature Loadings:\n", loadings)

# Can use a Loadings Heatmap to visual see how each predictor contributes to PC
plt.figure(figsize=(10,6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
plt.title("PCA Loadings Heatmap")
plt.xlabel("Principal Components")
plt.ylabel("Features")
plt.show()

# Scatterplots of Principal Components
import itertools

pairs = list(itertools.combinations(range(pca.n_components_), 2))

for pcx, pcy in pairs:
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca_full[:, pcx], X_pca_full[:, pcy],
                c=y, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.xlabel(f"PC{pcx+1}")
    plt.ylabel(f"PC{pcy+1}")
    plt.title(f"PCA Projection: PC{pcx+1} vs PC{pcy+1}")
    plt.show()

# Now  Let's Perform K-means clustering 
# features already scaled 

# Elbow Method
inertia = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_pca_full)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method")
plt.show()

# fit with chosen k of 9
kmeans_full = KMeans(n_clusters=9, random_state=42)
labels_full = kmeans_full.fit_predict(X_pca_full)

# Slihouette Score
score = silhouette_score(X_pca_full, labels_full)
print(f"Silhouette Score: {score:.3f}")

# Step 4: Visualize clusters
plt.scatter(X_pca_full[:,0], X_pca_full[:,1], c=labels_full, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_full.cluster_centers_[:,0], kmeans_full.cluster_centers_[:,1],
            s=250, c='red', marker='X', label='Centroids')
plt.title("Full K-Means Clustering with Centroids (Real Dataset)")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend()
plt.show()

# Let's run xgboost and random forest regression
# To Predict GFR

# Define Features
features = ["Creatinine", "BUN", "Urine_Output", "Diabetes", "Hypertension",
"Age", "Protein_in_Urine", "Water_Intake", "CKD_Status"]

X = df_2[features]
y = df_2['GFR']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 42
)

print(f"Size of training set, {X_train.shape[0]}")
print(f"Size of testing set, {X_test.shape[0]}")

# Initiate models
models = {
    "XGBoost": xgb.XGBRegressor(),
    "Random Forest": RandomForestRegressor()
}


# Train
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mean_square = mean_squared_error(y_test, preds)
    R2 = r2_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Mean Squared Error (MSE): {mean_square:.2f}")
    print(f"R2 score: {R2:.2f}")

# Let's Look at the feature importance for each
# For XGBoost and Random Forest
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'XGBoost': models["XGBoost"].feature_importances_,
    'Random Forest': models["Random Forest"].feature_importances_
}).sort_values(by="XGBoost", ascending=False)

# View feature importance for XGBoost and RandomForest
print("\nFeature Importance:\n", feat_importance)

# View feature importance for each on bar charts
for name, model in models.items():
    # Get feature importance
    importances = model.feature_importances_

    # Create series for plotting
    feat_series = pd.Series(importances, index=X_train.columns).sort_values(ascending=True)

    # Create figure
    plt.figure(figsize=(8,6))

    # Plot
    feat_series.plot(kind="barh", color="darkslategray")

    # Formatting
    plt.title(f"Feature Importance: {name}", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Display plot 
    plt.tight_layout()
    plt.show()

# CKD_Status is a very strong predictor of GFR. Actually it looks as if it can
# be the only predictor that we would need to use to predict GFR. Others could be removed from
# the model.


