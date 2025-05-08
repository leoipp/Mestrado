import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import joblib
import glob
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

#%% Load the Random Forest model
model = joblib.load(r'C:\Users\c0010261\Scripts\Mestrado\Data\RandomForestRegressor.pkl')
print('Model variables:')
print(model.feature_names)
print(type(model))

#%% Combine variables ['Elev variance', 'Elev P90', 'Elev CURT mean CUBE']
folder_variables = glob.glob(r'M:\4.Tabalhos_Especiais\4.33.LiDAR\Prognose\Predicao\2022\Validacao\Variaveis\ByTalhao\*.tif')

shapes = {}
for f in folder_variables:
    base = os.path.basename(f).rsplit('_', 1)[0]
    shapes.setdefault(base, []).append(f)

#%% Compute the predictions
ipc_data = pd.read_excel(r'C:\Users\c0010261\Scripts\Mestrado\Data\CNB_LIDAR_IPC_FBA.xlsx')
def read_raster(file_path):
    """Lê um arquivo raster e retorna o array numpy."""
    with rasterio.open(file_path) as src:
        return src.read(1)

# Exemplo de dicionário 'shapes' (adicione o seu)
# shapes = {'talhao_1': ['var1.tif', 'var2.tif', 'var3.tif'], ...}

for key, files in shapes.items():
    talhao_nome = key.replace("_", "-")
    idade_row = ipc_data[ipc_data['Talhao'] == talhao_nome]

    if idade_row.empty:
        print(f"[ERRO] Talhão '{talhao_nome}' não encontrado na planilha.")
        continue

    idade_valor = int(idade_row['Idade'].values[0])

    # Lê os rasters e empilha em 3D
    arrays = [read_raster(f) for f in files]
    stack = np.stack(arrays, axis=-1)

    height, width, n_bands = stack.shape
    n_pixels = height * width

    # Achata e insere a idade como última variável
    stack_2d = stack.reshape((n_pixels, n_bands))
    idade_array = np.full((n_pixels, 1), idade_valor, dtype=float)
    stack_2d_with_age = np.concatenate((stack_2d, idade_array), axis=1)

    # Máscara de valores válidos (sem NaN)
    valid_mask = ~np.any(np.isnan(stack_2d_with_age), axis=1)
    valid_data = stack_2d_with_age[valid_mask]

    if valid_data.shape[0] == 0:
        print(f"[ERRO] Talhão '{talhao_nome}' não possui pixels válidos para predição.")
        continue

    # Previsão por árvore
    all_tree_preds = np.stack([tree.predict(valid_data) for tree in model.estimators_], axis=0)

    # Média e desvio padrão das árvores
    mean_pred = np.mean(all_tree_preds, axis=0)
    std_pred = np.std(all_tree_preds, axis=0)

    # Cria arrays completos com NaNs
    predicted_full = np.full((n_pixels,), np.nan)
    uncertainty_full = np.full((n_pixels,), np.nan)

    # Insere resultados nas posições válidas
    predicted_full[valid_mask] = mean_pred
    uncertainty_full[valid_mask] = std_pred

    # Reshape para formato raster (altura, largura)
    predicted_raster = predicted_full.reshape((height, width))
    uncertainty_raster = uncertainty_full.reshape((height, width))

    # Salva os arquivos raster .tif
    with rasterio.open(files[0]) as src_ref:
        profile = src_ref.profile
        profile.update(dtype='float32', count=1, nodata=np.nan)

        output_dir = r"C:\Users\c0010261\Scripts\Mestrado\Data\Estimated"
        os.makedirs(output_dir, exist_ok=True)

        pred_path = os.path.join(output_dir, f"{talhao_nome}_estimado.tif")
        unc_path = os.path.join(output_dir, f"{talhao_nome}_incerteza.tif")

        with rasterio.open(pred_path, 'w', **profile) as dst:
            dst.write(predicted_raster.astype('float32'), 1)

        with rasterio.open(unc_path, 'w', **profile) as dst:
            dst.write(uncertainty_raster.astype('float32'), 1)

    # Atualiza DataFrame com valor médio estimado
    estimated_mean = np.nanmean(predicted_raster)
    ipc_data.loc[ipc_data['Talhao'] == talhao_nome, 'ESTIMADO'] = estimated_mean

    print(f"[OK] Talhão '{talhao_nome}': estimado = {estimated_mean:.2f}")
ipc_data['RESIDUAL'] = (ipc_data['ESTIMADO']-ipc_data['Prod IPC'])/ipc_data['Prod IPC']
#%% Plot graphs
fig, axis = plt.subplots(1, 3, figsize=(12, 6))

axis[0].scatter(ipc_data['Prod IPC'], ipc_data['ESTIMADO'], alpha=.7)
axis[0].plot([ipc_data['Prod IPC'].min(), ipc_data['Prod IPC'].max()], [ipc_data['Prod IPC'].min(), ipc_data['Prod IPC'].max()], color='red', linestyle='--')
axis[0].set_xlabel('Volume total com casca (m³/ha) - Obs.')
axis[0].set_ylabel('Volume total com casca (m³/ha) - Est.')
axis[0].set_xlim(0, 600)
axis[0].set_ylim(0,600)
#axis[0].set_title(f'VTCC - Observed vs Estimated {ipc_data["RESIDUAL"].mean():.2%}')
axis[0].grid(linestyle='--', alpha=0.5, color='grey')

axis[1].scatter(ipc_data['ESTIMADO'], ipc_data['RESIDUAL'], alpha=.7)
axis[1].axhline(y=0, color='red', linestyle='--')
axis[1].set_xlabel('Volume total com casca (m³/ha) - Est.')
axis[1].set_ylabel('Resíduos')
#axis[1].set_title('Residuals vs VTCC Est.')
axis[1].set_ylim(-1, 1)
axis[1].grid(linestyle='--', alpha=0.5, color='grey')
sns.histplot(
    ipc_data['RESIDUAL'],
    bins=30,
    color='#1f77b4',
    edgecolor='#1f77b4',
    linewidth=0.8,
    alpha=0.7,
    ax=axis[2],
    kde=True
)
axis[2].set_xlabel('Resíduos')
axis[2].set_ylabel('Frequência')
#axis[2].set_title('Residuals Distribution')
axis[2].set_xlim(-1, 1)
axis[2].grid(linestyle='--', alpha=0.5, color='grey')
axis[2].axvline(x=0, color='red', linestyle='--')
axis[2].set_xticks(np.arange(-1, 1.1, 0.2))

plt.tight_layout()
plt.show()

#%% Predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

r2_test = r2_score(ipc_data['Prod IPC'], ipc_data['ESTIMADO'])
print(f"Test R²: {r2_test:.2f}")

# RMSE
rmse_train = np.sqrt(mean_squared_error(ipc_data['Prod IPC'], ipc_data['ESTIMADO']))
print(f"Train RMSE: {rmse_train:.2f}")

# MAE
mae_test = mean_absolute_error(ipc_data['Prod IPC'], ipc_data['ESTIMADO'])
print(f"Test MAE: {mae_test:.2f}")

# Residuals
med = np.mean(ipc_data['RESIDUAL']) * 100
print(f"Test Residuals: {med:.2f}%")

#%% Save the results
ipc_data.to_excel(r'C:\Users\c0010261\Scripts\Mestrado\Data\Conferencia_IFPC_Predictions.xlsx', index=False)

#%% Validate random in raster
# ex1 = np.array([121.578499, 26.097700, 18.420601, 98]).reshape(1, -1)
# pred = model.predict(ex1)
# print(pred)