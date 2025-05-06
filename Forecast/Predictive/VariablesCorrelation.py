import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Read the data
df = pd.read_excel(r"G:\PycharmProjects\Mestrado\Data\IFC_LiDAR_Plots_RTK_Cleaned.xlsx")
print(df.shape)
#%% Calculate Pearson correlation with VTCC
numeric_df = df.select_dtypes(include='number')
correlations = numeric_df.corr()['VTCC(m³/ha)'].drop('VTCC(m³/ha)') .sort_values(ascending=False)
print(correlations)
#%% Plot the correlations
plt.figure(figsize=(10, 6))
plt.hist(correlations, bins=30, alpha=0.7, color='blue', zorder=2)
plt.title('VTCC Correlation With VTCC')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Frequency')
plt.grid(linestyle='--', color='gray', alpha=0.5, zorder=1)
plt.show()
#%% Choose the correlations +-.6
correlations = correlations[abs(correlations) > 0.6]
# Remove inventory variables
correlations = correlations[correlations.index.str.contains('Elev')]
print(correlations)
#%% Plot the correlations with a heatmap
subset = df[correlations.index.tolist() + ['VTCC(m³/ha)', 'Idade (meses)']]
plt.figure(figsize=(10, 8))
sns.heatmap(subset.corr(), cmap="YlGnBu", annot=True, mask=np.triu(np.ones_like(subset.corr())), square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()
