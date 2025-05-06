import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% Dataframe reader
df = pd.ExcelFile(r"G:\PycharmProjects\Mestrado\Data\IFC_LiDAR_Plots_RTK.xlsx").parse("IFC - Plots Consistido")
print(df.shape)
#%% Scatter plot of VTCC vs Age (months)
plt.figure(figsize=(10, 6))
plt.scatter(df['Idade (meses)'], df['VTCC(m³/ha)'], alpha=0.7)
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.title('VTCC vs Age (months)')
plt.xlabel('Age (months)')
plt.ylabel('VTCC (m³/ha)')
plt.show()
#%% Histogram of VTCC and Age
fig, axis = plt.subplots(1, 2, figsize=(10, 6))

axis[0].hist(df['VTCC(m³/ha)'], bins=30, alpha=0.7, color='blue', zorder=2)
axis[0].grid(linestyle='--', color='gray', alpha=0.5, zorder=1)
axis[0].set_title('Histogram of VTCC')
axis[0].set_xlabel('VTCC (m³/ha)')
axis[0].set_ylabel('Frequency')

axis[1].hist(df['Idade (meses)'], bins=30, alpha=0.7, color='blue', zorder=2)
axis[1].grid(linestyle='--', color='gray', alpha=0.5, zorder=1)
axis[1].set_title('Histogram of VTCC')
axis[1].set_xlabel('Age (months)')
axis[1].set_ylabel('Frequency')
plt.show()
#%% Boxplot of VTCC by Age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Idade (meses)', y='VTCC(m³/ha)', data=df, palette="Set2")
plt.title('Boxplot of VTCC by Age (months)')
plt.xlabel('Age (months)')
plt.ylabel('VTCC (m³/ha)')
plt.xticks(rotation=45)
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.show()
#%% Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])
#%% Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

#%% Check for negative values in VTCC
negative_vtcc = df[df['VTCC(m³/ha)'] < 0]
if not negative_vtcc.empty:
    print(f"Negative VTCC values found in {len(negative_vtcc)} rows.")
    df = df[df['VTCC(m³/ha)'] >= 0]
    print("Negative VTCC values removed.")

#%% Check for negative values in Age
negative_age = df[df['Idade (meses)'] < 0]
if not negative_age.empty:
    print(f"Negative Age values found in {len(negative_age)} rows.")
    df = df[df['Idade (meses)'] >= 0]
    print("Negative Age values removed.")
#%% Check for age values greater than 200 months
age_greater_200 = df[df['Idade (meses)'] > 200]
if not age_greater_200.empty:
    print(f"Age values greater than 200 months found in {len(age_greater_200)} rows.")
    df = df[df['Idade (meses)'] <= 200]
    print("Rows with Age greater than 200 months removed.")
#%% Check for outliers using IQR (NOT RECOMMENDED - CHOOSE ZSCORE)
Q1 = df['VTCC(m³/ha)'].quantile(0.25)
Q3 = df['VTCC(m³/ha)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['VTCC(m³/ha)'] < lower_bound) | (df['VTCC(m³/ha)'] > upper_bound)]
print(f"Number of outliers in VTCC: {len(outliers)}")
if not outliers.empty:
    print("Outliers found:")
    print(outliers)
    #df = df[~df.index.isin(outliers.index)]
    print("Outliers removed.")
plt.figure(figsize=(10, 6))
plt.scatter(df['Idade (meses)'], df['VTCC(m³/ha)'], alpha=0.7)
plt.scatter(outliers['Idade (meses)'], outliers['VTCC(m³/ha)'], color='red', label='Outliers', alpha=0.7)
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.title('VTCC vs Age (months) with Outliers')
plt.xlabel('Age (months)')
plt.ylabel('VTCC (m³/ha)')
plt.legend()
plt.show()
#%% Check for outliers using Z-score
from scipy import stats
z_scores = stats.zscore(df['VTCC(m³/ha)'])
abs_z_scores = np.abs(z_scores)
threshold = 3
outliers_z = df[abs_z_scores > threshold]
print(f"Number of outliers in VTCC using Z-score: {len(outliers_z)}")
if not outliers_z.empty:
    print("Outliers found using Z-score:")
    print(outliers_z)
    df = df[~df.index.isin(outliers_z.index)]
    print("Outliers removed.")
plt.figure(figsize=(10, 6))
plt.scatter(df['Idade (meses)'], df['VTCC(m³/ha)'], alpha=0.7)
plt.scatter(outliers_z['Idade (meses)'], outliers_z['VTCC(m³/ha)'], color='red', label='Outliers Z-score', alpha=0.7)
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.title('VTCC vs Age (months) with Outliers Z-score')
plt.xlabel('Age (months)')
plt.ylabel('VTCC (m³/ha)')
plt.legend()
plt.show()
#%% Check for extreme lower values - not removed with Z-score
extreme_lower = df[(df["Idade (meses)"] >= 50) & (df['VTCC(m³/ha)'] < 100)]
print(f"Extreme lower values found in {len(extreme_lower)} rows.")
if not extreme_lower.empty:
    print("Extreme lower values found:")
    print(extreme_lower)
    df = df[~df.index.isin(extreme_lower.index)]
    print("Extreme lower values removed.")
plt.figure(figsize=(10, 6))
plt.scatter(df['Idade (meses)'], df['VTCC(m³/ha)'], alpha=0.7)
plt.scatter(extreme_lower['Idade (meses)'], extreme_lower['VTCC(m³/ha)'], color='red', label='Extreme Lower Values', alpha=0.7)
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.title('VTCC vs Age (months) with Extreme Lower Values')
plt.xlabel('Age (months)')
plt.ylabel('VTCC (m³/ha)')
plt.legend()
plt.show()
