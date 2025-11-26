# This dataset looks into exploring the relatonship
# between social media usage and mental health
# variables include: User_ID, Age, Gender, DailyScreenTime,
# SleepQuality, StressLevel, DaysW/OSocialMedia,ExerciseFrequency,
# SocialMediaPlatform and HappinessIndex

# Purpose: Exploratory, Correlation, PCA, and Factor Analysis

# Load Libraries
library(tidyverse)
library(psych)
library(corrr)
library(ggcorrplot)
library(FactoMineR)
library(factoextra)
library(corrplot)
library(GGally)
library(plotly)
library(aplpack)
library(scatterplot3d)
library(asbio)
library(ggfortify)
library(ggrepel)


# Set working directory
setwd("C:/Users/shirl/OneDrive/Desktop/R")

# Load data
df <- read.csv("Mental_Health_and_Social_Media_Balance_Dataset.csv", stringsAsFactors = TRUE)

# Dataset consists of 500 observations with 10 variables

# view first few rows
head(df)

# Rename columns for simplicity
df <- df %>%
  rename(
    Screen_Time      = Daily_Screen_Time.hrs.,
    Sleep_Quality    = Sleep_Quality.1.10.,
    Stress_Level     = Stress_Level.1.10.,
    Days_No_Social   = Days_Without_Social_Media,
    Exercise_Freq    = Exercise_Frequency.week.,
    Social_Media     = Social_Media_Platform,
    Happiness_Level  = Happiness_Index.1.10.
  )

# Check contents of data
str(df)

# Look at summary statistics for data
summary(df)


# Check for missing values 
colSums(is.na(df))
# There are no missing values


# Check to see if there are any duplicate values
sum(duplicated(df))
# No duplicates in dataset

# Analyze numerical variables
num_vars <- df %>% select(where(is.numeric))

# View histograms for numeric variables
if(ncol(num_vars) > 0) {
  num_vars %>%
    gather(key="variable", value="value") %>%
    ggplot(aes(x=value)) +
    geom_histogram(fill="blue", color="black", bins=15) +
    facet_wrap(~variable, scales="free") +
    theme_minimal() +
    labs(title="Distribution of Numeric Variables")
}

# Look at count and bar plots for categorical variables
cat_vars <- df %>% 
  select(where(is.factor)) %>%
  select(-User_ID)

# count and bar plots
if(ncol(cat_vars) > 0){
  for(v in names(cat_vars)){
    print(df %>% group_by(.data[[v]]) %>% summarise(count = n()))
    
    p <- ggplot(df, aes_string(x=v, fill=v)) +
      geom_bar() +
      theme_minimal() +
      labs(title=paste("Count of", v))
      
    print(p)
  }
}

#### Outlier Detection ####
# Let's check to see if there are outliers by looking at boxplot of numeric variables
if(ncol(num_vars) > 0){
  num_vars %>%
    pivot_longer(cols = everything()) %>%
    ggplot(aes(x = name, y = value, fill = name)) + 
    geom_boxplot(outlier.colour = "red", alpha = 0.6) +
    labs(title = "Boxplots of Numeric Variables", x = "Variable", y = "Value") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = "none")
}

# Daily Screen Time has 1 outlier
# Exercise frequency Week has 1 outlier and Stress Level has 1 outlier

## Let's apply Mahalanobis Distance
df_2 <- num_vars

# Standardize data
df_scaled <- scale(df_2)

# Compute Mahlanobis Distance
center <- colMeans(df_scaled)
cov_matrix <- cov(df_scaled)
mahal <- mahalanobis(df_scaled, center, cov_matrix)

# Determine outliers
threshold <- qchisq(0.975, df = ncol(df_scaled))
outlier_flag <- mahal > threshold
table(outlier_flag)

# Dataframe for plotting
df_plot <- data.frame(
  ID = df$User_ID,
  Index = 1:nrow(df_scaled),
  Mahalanobis_Dis = mahal,
  Outlier = outlier_flag
)

# Visualize
outliers <- df_plot[df_plot$Outlier,]

ggplot(df_plot, aes(x = Index, y = Mahalanobis_Dis, color = Outlier)) +
  geom_point(size = 2) +
  geom_hline(yintercept = threshold, linetype="dashed") +
  geom_text_repel(data = outliers, aes(label = ID), size = 3) +
  labs(title="Mahalanobis Distance Outliers with User IDs",
       x = "Observation Index", y="Mahalanobis Distance") +
  theme_minimal()

# 11 Users indentifies as Outliers: U040, U013, U011, U095, U249, U153, U216,
# U264, U450, U494, U490

### Check correlation between numeric variables
# Get correlation matrix
cor_matrix <- cor(num_vars)

# Look at ScatterPlot Matrix
pairs(num_vars)

# Correlation heatmap
ggcorrplot(cor_matrix, lab = TRUE, hc.order = TRUE, title = "Correlation Heatmap")

# Happiness Index and Daily Screen Time = -0.71 (strong negative correlation)
# Ex: As Happiness increases daily screen time decreases

# Sleep Quality and Daily Screen Time = -0.76 (Strong negative correlation)
# As sleep quality increases, daily screen time decreases

# Stress Level and Daily Screen Time = 0.74 (Strong positive correlation)
# As stress level increases, daily screen time increases

# Happiness and Sleep Quality = 0.68 (strong positive correlation)
# As happiness increases, sleep quality increases also

# Happiness and Stress Level = -0.74 (strong negative correlation)
# as happiness increases, stress levels decreases

# Sleep Quality and Stress Level = - 0.58 (Moderately strong positive correlation)
# As sleep quality increases, stress levels decreases

# There is correlation among the variables but majority are not correlated
# Sleep, Stress and Screen Time have high amount of impact
# Age does not correlate much with any of the variables

# Dataset has multicollinearity since more than two variables have correlations |r| > 7.


### PRINCIPAL COMPONENT ANALYSIS
df_pc <- num_vars

# Since we have variables that are highly correlated let's use prcomp for PCA
df_pc_result <- prcomp(df_pc, scale. = TRUE)

# View PCA
summary(df_pc_result)

# PC1 explains 44.46% of variance, PC2 15.32%, PC3 14.25%, PC4 13.34% => 87.36% of total variance in dataset
# Looks like 4 Principal components works

# Let's look a the eigenvalues and a scree plot to determine the necessary principal components

# access loadings
loadings <- df_pc_result$rotation
print(loadings)

# PC1: -0.5172 (Screen Time), 0.4850 (Sleep Quality), -0.4914 (Stress Level), Happiness (0.5020)
# PC2: 0.709390 (Age), 0.654763 (Exercise_Frequency)
# PC3: -0.937681 (Days without Social Media)



# Extract standard deviation 
sdev <- df_pc_result$sdev
# calculate eigenvalues
eigenvalues <- sdev^2
# print eigenvalues
print(eigenvalues)

# Based on Kisen Criterion: eigenvalue > 1
# PC1 = 3.11, PC2 = 1.07, PC3 = 0.99

# Lets' look at scree plot
fviz_eig(df_pc_result, addlabes = TRUE, barfill = "blue", main = "Scree Plot - PCA")
# Scree plot looks as if 4 Principal Components work 

# Let's look at seperate plots for users and variables since biplot will be too clustered
fviz_pca_ind(df_pc_result,
             repel = TRUE, # avoids overlapping
             col.ind = "blue", # colors for users
             alpha.ind = 0.6, # transparency for points
             title = "PCA - Users"
             )

# Plot for variables
fviz_pca_var(df_pc_result, 
             repel = TRUE, 
             col.var = "red", 
             title = "PCA - Variables")


# Create DataFrame of PC's for plots
pc_df <- as.data.frame(df_pc_result$x[, 1:4]) %>%
  mutate(ID = df$User_ID)

# Let's look at Scatterplot for each PC
# Only include the first 4 PCs
ggpairs(pc_df[, 1:4],
        upper = list(continuous = wrap("points", alpha = 0.6, size = 1.5)),
        lower = list(continuous = wrap("points", alpha = 0.6, size = 1.5)),
        diag = list(continuous = "densityDiag"))


# Now let's see if there are any unobservables factors that influence variables
df_fa <- num_vars

# Determine the number of factors
fa1 <- factanal(df_fa, factors = 1, rotation = "varimax")
print(fa1)
# p-value is extremely small try 2

fa2 <- factanal(df_fa, factors = 2, rotation = "varimax")
print(fa2)
# p-value is 0.0233, still small so let's try 3

fa3 <- factanal(df_fa, factors = 3, rotation = "varimax")
print(fa3)
# p-value is 0.128, seems significant enough, let's use 3 factors

# Choose 3 factor model
names(df_fa) <- c("Stress", "Happiness", "Screen", "Sleep", "NoSocial", "Exercise", "Age")
fa_results <- psych::fa(r = df_fa, nfactors = 3, rotate = "varimax")
fa.diagram(fa_results)

# Add factors scores to dataframe
fa.scores <- fa_results$scores
df_full_fa <- bind_cols(df_fa, as.data.frame(fa.scores))
