import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
import joblib
import glob


#%% Load the Random Forest model
model = joblib.load(r'G:\PycharmProjects\Mestrado\Forecast\Predictive\Models\RandomForestRegressor.pkl')
print('Model variables:')
print(model.feature_names)
print(type(model))

#%% Combine variables ['Elev variance', 'Elev P90', 'Elev CURT mean CUBE']
folder_variables = glob.glob(r'G:\PycharmProjects\Mestrado\Data\Predictive\*.tif')

shapes = {}
for f in folder_variables:
    base = os.path.basename(f).rsplit('_', 1)[0]
    shapes.setdefault(base, []).append(f)

#%% Compute the predictions
ipc_data = pd.read_excel(r'G:\PycharmProjects\Mestrado\Data\DataFrames\Conferencia_IFPC.xlsx')
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

        output_dir = r"G:\PycharmProjects\Mestrado\Data\Predictive\TESTE"
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

#%% Plot graphs
ipc_data['RESIDUAL'] = (ipc_data['ESTIMADO']-ipc_data['Prod IPC'])/ipc_data['Prod IPC']
fig, axis = plt.subplots(1, 3, figsize=(12, 6))

axis[0].scatter(ipc_data['Prod IPC'], ipc_data['ESTIMADO'], alpha=.7)
axis[0].plot([ipc_data['Prod IPC'].min(), ipc_data['Prod IPC'].max()], [ipc_data['Prod IPC'].min(), ipc_data['Prod IPC'].max()], color='red', linestyle='--')
axis[0].set_xlabel('VTCC Obs.')
axis[0].set_ylabel('VTCC Est.')
axis[0].set_title(f'VTCC - Observed vs Estimated {ipc_data["RESIDUAL"].mean():.2%}')
axis[0].grid(linestyle='--', alpha=0.5, color='grey')

axis[1].scatter(ipc_data['ESTIMADO'], ipc_data['RESIDUAL'], alpha=.7)
axis[1].axhline(y=0, color='red', linestyle='--')
axis[1].set_xlabel('VTCC Est.')
axis[1].set_ylabel('Residuals')
axis[1].set_title('Residuals vs VTCC Est.')
axis[1].set_ylim(-1, 1)
axis[1].grid(linestyle='--', alpha=0.5, color='grey')

axis[2].hist(ipc_data['RESIDUAL'], bins=20, alpha=.7)
axis[2].set_xlabel('Residuals')
axis[2].set_ylabel('Frequency')
axis[2].set_title('Residuals Distribution')
axis[2].set_xlim(-1, 1)
axis[2].grid(linestyle='--', alpha=0.5, color='grey')
axis[2].axvline(x=0, color='red', linestyle='--')
axis[2].set_xticks(np.arange(-1, 1.1, 0.2))

plt.tight_layout()
plt.show()

#%% Save the results
ipc_data.to_excel(r'G:\PycharmProjects\Mestrado\Data\DataFrames\Conferencia_IFPC_Predictions.xlsx', index=False)

#%% Validate random in raster
# ex1 = np.array([121.578499, 26.097700, 18.420601, 98]).reshape(1, -1)
# pred = model.predict(ex1)
# print(pred)