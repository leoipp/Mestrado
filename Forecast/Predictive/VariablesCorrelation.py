import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%% Read the data
df = pd.read_excel(r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned.xlsx")
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
#%% Recursive feature elimination - RFE
feature_cols = df.columns[df.columns.str.contains('Elev')]
X = df[feature_cols].values
Y = df[['VTCC(m³/ha)']].values.ravel()
# Cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=100)
hyper_params = [{'n_features_to_select': list(range(1, X.shape[1] + 1))}]
# Base model
model = RandomForestRegressor()
# RFE and GridSearchCV
rfe = RFE(estimator=model)
model_cv = GridSearchCV(estimator=rfe, param_grid=hyper_params,
                        scoring='neg_mean_squared_error',
                        cv=folds, verbose=1, return_train_score=True)
model_cv.fit(X, Y)
# Features ranking
print("\nRanking das features (ordenado):")
ranking_sorted = sorted(zip(feature_cols,
                            model_cv.best_estimator_.ranking_,
                            model_cv.best_estimator_.support_),
                        key=lambda x: x[1])
for i, (name, rank, selected) in enumerate(ranking_sorted):
    print(f"{i+1:02d}. {name:20s} | Rank: {rank:2d} | Selected: {selected}")
#%% Plot the correlations with a heatmap
selected_cols = feature_cols[model_cv.best_estimator_.support_]
corr_matrix = df[selected_cols].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 5},  # reduz o tamanho da fonte
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
    square=True,
    cbar_kws={"shrink": .8}
)
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Heatmap')
plt.show()
#%% Variance Inflation Factor (VIF)
X_sel = df[selected_cols]
vif_data = pd.DataFrame()
vif_data['Feature'] = X_sel.columns
vif_data['VIF'] = [variance_inflation_factor(X_sel.values, i) for i in range(X_sel.shape[1])]
vif_data.to_excel(r".\Data\VIF.xlsx", index=False)
print(vif_data.sort_values(by='VIF', ascending=False))
#%% Assuming variables of chosen (Elev P90, Elev variance, Elev CURT mean CUBE, Idade (meses))
X_sel = df[['Elev P90', 'Elev variance', 'Elev CURT mean CUBE', 'Idade (meses)']]
vif_data = pd.DataFrame()
vif_data['Feature'] = X_sel.columns
vif_data['VIF'] = [variance_inflation_factor(X_sel.values, i) for i in range(X_sel.shape[1])]
#vif_data.to_excel(r"G:\PycharmProjects\Mestrado\Data\VIF.xlsx", index=False)
print(vif_data.sort_values(by='VIF', ascending=False))
#%% Plot the correlations with a heatmap of chosen variables (Elev P90, Elev variance, Elev CURT mean CUBE, Idade (meses))
corr_matrix = df[['Elev P90', 'Elev variance', 'Elev CURT mean CUBE', 'Idade (meses)']].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
    square=True,
    cbar_kws={"shrink": .8}
)
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Heatmap')
plt.show()
