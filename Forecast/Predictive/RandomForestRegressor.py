import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

#%% Read the data
df = pd.read_excel(r"G:\PycharmProjects\Mestrado\Data\IFC_LiDAR_Plots_RTK_Cleaned.xlsx")
print(df.shape)
#%% Define variables chosen (ELEV P90, ELEV VARIANCE, ELEV CURT MEAN CUBE, AGE(MONTHS))
feature_names = ['Elev P90', 'Elev variance', 'Elev CURT mean CUBE', 'Idade (meses)']
subset = df[feature_names + ['VTCC(m³/ha)']].copy()
print(subset.shape)
#%% Variables distribution - Organize data
X = subset[feature_names].values.astype(float)
y = subset['VTCC(m³/ha)'].values.astype(float)
plt.figure(figsize=(15, 5))
# Plot each variable's distribution
for i, column in enumerate(feature_names):
    plt.subplot(1, len(feature_names), i + 1)  # Create a subplot for each variable
    sns.histplot(X[:, i], bins=30, kde=True)  # Histogram with kernel density estimate
    plt.grid(linestyle='--', color='gray', alpha=0.5)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
#%% Visualize pairplot of chosen variables
sns.pairplot(subset, diag_kind="kde")
plt.grid(linestyle='--', color='gray', alpha=0.5)
plt.show()
#%% Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = y_train.flatten()
y_test = y_test.flatten()
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
#%% Random Forest Regressor - Building Hyperparameter Grid for Randomized Search CV
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
# Define the hyperparameter grid
# Trees in the forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=20)]
# Variables number of features to consider when looking for the best split
max_features = ['sqrt', 'log2', 1, 0.2, 0.3, 0.4, 0.5, 0.8]
# Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(10, 150, num=10)]
max_depth.append(None)
# Minimum number of samples to split an internal node
min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Minimum number of samples required to be at a leaf node
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Bootstrap samples when building trees
bootstrap = [True, False]
# Hyperparameter grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
#%% Random Forest Regressor - Randomized Search CV - TUNNING
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(x_train, y_train)
print("Optimum params:")
print(rf_random.best_params_)
#%% Random Forest Regressor Adjusted - OPTIMUM PARAMS
# Option 1: Pass the best parameters directly using ** (unpacking)
rfReg = RandomForestRegressor(**rf_random.best_params_, random_state=42)
rfReg.fit(x_train, y_train)
# Alternatively you can use the best_estimator_ directly:
# rfReg = rf_random.best_estimator_
#%% Model Evaluation - Scores
# Predictions
y_pred_test = rfReg.predict(x_test)
y_pred_train = rfReg.predict(x_train)
# Statistical metrics
# R²
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
print(f"Train R²: {r2_train:.2f}")
print(f"Test R²: {r2_test:.2f}")
# RMSE
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
# MAE
mae_test = np.mean(np.abs(y_test - y_pred_test))
mae_train = np.mean(np.abs(y_train - y_pred_train))
print(f"Train MAE: {mae_train:.2f}")
print(f"Test MAE: {mae_test:.2f}")
# Residuals
residuals_test = (y_test - y_pred_test)/y_test
residuals_train = (y_train - y_pred_train)/y_train
print(f"Train Residuals: {np.mean(residuals_train):.2f}%")
print(f"Test Residuals: {np.mean(residuals_test):.2f}%")
#%% Observed vs Predicted - Statistical metric plots
fig, axis = plt.subplots(2, 3, figsize=(8,6))
# Test - Observed vs Predicted
axis[0,0].scatter(y_test, y_pred_test, alpha=0.7)
axis[0,0].set_xlabel("VTCC (m³/ha) - Obs.")
axis[0,0].set_ylabel("VTCC (m³/ha) - Est.")
axis[0,0].set_title("Comparison of Observed vs Predicted Values")
axis[0,0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
axis[0,0].grid(linestyle='--', color='gray', alpha=0.5)
# Test - Residuals
axis[0,1].scatter(y_pred_test, residuals_test, alpha=0.7)
axis[0,1].set_xlabel("VTCC (m³/ha) - Est.")
axis[0,1].set_ylabel("Residuals")
axis[0,1].set_title("Residuals vs Predicted Values")
axis[0,1].axhline(y=0, color='red', linestyle='--')
axis[0,1].set_ylim(-1, 1)
axis[0,1].grid(linestyle='--', color='gray', alpha=0.5)
# Test - Residual hist
sns.histplot(residuals_test, bins=30, color='steelblue', alpha=0.7, ax=axis[0, 2], kde=True)
axis[0,2].set_xlim(-1,1)
axis[0,2].set_title("Residuals Distribution")
axis[0,2].set_xlabel("Residuals")
axis[0,2].set_ylabel("Frequency")
axis[0,2].axvline(x=0, color='red', linestyle='--')
axis[0,2].grid(linestyle='--', alpha=0.5, color='grey')
# Train - Observed vs Predicted
axis[1,0].scatter(y_train, y_pred_train, alpha=0.7)
axis[1,0].set_xlabel("VTCC (m³/ha) - Obs.")
axis[1,0].set_ylabel("VTCC (m³/ha) - Est.")
axis[1,0].set_title("Comparison of Observed vs Predicted Values")
axis[1,0].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
axis[1,0].grid(linestyle='--', color='gray', alpha=0.5)
# Train - Residuals
axis[1,1].scatter(y_pred_train, residuals_train, alpha=0.7)
axis[1,1].set_xlabel("VTCC (m³/ha) - Est.")
axis[1,1].set_ylabel("Residuals")
axis[1,1].set_title("Residuals vs Predicted Values")
axis[1,1].axhline(y=0, color='red', linestyle='--')
axis[1,1].set_ylim(-1, 1)
axis[1,1].grid(linestyle='--', color='gray', alpha=0.5)
# Train - Residual hist
sns.histplot(residuals_train, bins=30, color='steelblue', alpha=0.7, ax=axis[1, 2], kde=True)
axis[1,2].set_xlim(-1,1)
axis[1,2].set_title("Residuals Distribution")
axis[1,2].set_xlabel("Residuals")
axis[1,2].set_ylabel("Frequency")
axis[1,2].axvline(x=0, color='red', linestyle='--')
axis[1,2].grid(linestyle='--', alpha=0.5, color='grey')
# Adjust layout
plt.suptitle("Random Forest Regressor - Statistical Metrics", fontsize=16)
plt.tight_layout()
plt.show()
#%% Feature Importance
importances = rfReg.feature_importances_
indices = np.argsort(importances)
# Print the feature ranking
print("Feature ranking:")
for f in range(len(feature_names)):
    print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.3f})")
# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(feature_names)), importances[indices], align="center", color='steelblue')
plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
plt.xlabel("Importance")
plt.grid(linestyle='--', alpha=0.5, color='grey')
plt.tight_layout()
plt.show()
#%% Insert predictions into the original dataframe
df['Predicted VTCC(m³/ha)'] = rfReg.predict(X)
df['Residuals'] = (df['VTCC(m³/ha)'] - df['Predicted VTCC(m³/ha)'])/df['VTCC(m³/ha)']
df.to_excel(r"G:\PycharmProjects\Mestrado\Data\IFC_LiDAR_Plots_RTK_Cleaned_Predictions.xlsx", index=False)
#%% Save the model
rfReg.feature_names = feature_names
joblib.dump(rfReg, r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Models\RandomForestRegressor.pkl")
