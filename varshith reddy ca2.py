# ==========================================
# MENTAL HEALTH DATA ANALYSIS PROJECT
# Using NumPy, Pandas, Matplotlib, Seaborn
# ==========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Plot Style
sns.set(style="whitegrid")

# Load Dataset
df = pd.read_csv(r"C:\Users\yanna\OneDrive\Documents\mental_health.csv")

# ==========================================
# BASIC DATASET INFORMATION
# ==========================================
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# DATA CLEANING
# ==========================================

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# Fill Missing Numeric Values with Median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill Missing Categorical Values with Mode
cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# VISUALIZATION 1: Age Distribution
# ==========================================
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# ==========================================
# VISUALIZATION 2: Gender Distribution
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

# ==========================================
# VISUALIZATION 3: Occupation Distribution
# ==========================================
plt.figure(figsize=(10,5))
sns.countplot(x='Occupation', data=df)
plt.title("Occupation Distribution")
plt.xticks(rotation=45)
plt.show()

# ==========================================
# VISUALIZATION 4: Stress Level by Occupation
# ==========================================
plt.figure(figsize=(10,5))
sns.boxplot(x='Occupation', y='Stress_Level', data=df)
plt.title("Stress Level by Occupation")
plt.xticks(rotation=45)
plt.show()

# ==========================================
# VISUALIZATION 5: Sleep Hours vs Stress Level
# ==========================================
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Sleep_Hours',
    y='Stress_Level',
    hue='Gender',
    data=df
)
plt.title("Sleep Hours vs Stress Level")
plt.show()

# ==========================================
# VISUALIZATION 6: Depression Cases
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(x='Depression', data=df)
plt.title("Depression Cases")
plt.xticks([0,1], ['No','Yes'])
plt.show()

# ==========================================
# VISUALIZATION 7: Anxiety Cases
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(x='Anxiety', data=df)
plt.title("Anxiety Cases")
plt.xticks([0,1], ['No','Yes'])
plt.show()

# ==========================================
# VISUALIZATION 8: Burnout Cases
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(x='Burnout', data=df)
plt.title("Burnout Cases")
plt.xticks([0,1], ['No','Yes'])
plt.show()

# ==========================================
# VISUALIZATION 9: Screen Time vs Depression
# ==========================================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Depression',
    y='Daily_Screen_Time',
    data=df
)
plt.title("Screen Time vs Depression")
plt.xticks([0,1], ['No Depression','Depression'])
plt.show()

# ==========================================
# VISUALIZATION 10: Physical Activity vs Stress
# ==========================================
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Physical_Activity',
    y='Stress_Level',
    data=df
)
plt.title("Physical Activity vs Stress Level")
plt.show()

# ==========================================
# VISUALIZATION 11: Lifestyle Habits Pie Charts
# ==========================================
fig, ax = plt.subplots(1, 3, figsize=(15,5))

df['Smoking'].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=ax[0],
    title='Smoking'
)

df['Alcohol'].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=ax[1],
    title='Alcohol'
)

df['Caffeine_Intake'].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=ax[2],
    title='Caffeine Intake'
)

plt.tight_layout()
plt.show()

# ==========================================
# VISUALIZATION 12: Correlation Heatmap
# ==========================================
plt.figure(figsize=(12,8))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.show()
