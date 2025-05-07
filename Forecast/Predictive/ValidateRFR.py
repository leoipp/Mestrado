import os.path

import numpy as np
import pandas as pd
from PIL import Image
import joblib
import glob

#%% Load the Random Forest model
model = joblib.load(r'.\Data\RandomForestRegressor.pkl')
print('Model variables:')
print(model.feature_names)

#%% Combine variables ['Elev P90', 'Elev variance', 'Elev CURT mean CUBE']
folder_variables = glob.glob(r'M:\4.Tabalhos_Especiais\4.33.LiDAR\Prognose\Predicao\2022\Validacao\Variaveis\ByTalhao\*.tif')
# Agrupar por base name (antes do último "_")
shapes = {}
for f in folder_variables:
    base = os.path.basename(f).rsplit('_', 1)[0]  # separa por "_" e pega tudo antes do último
    shapes.setdefault(base, []).append(f)
#%% Compute the predictions
idade = pd.read_excel(r'C:\Users\c0010261\Scripts\Mestrado\Data\CNB_LIDAR_IPC_FBA.xlsx')

for key, files in shapes.items():
    # Padroniza o nome do talhão
    talhao_nome = key.replace("_", "-")

    # Filtra a idade correspondente
    idade_row = idade[idade['Talhao'] == talhao_nome]

    if not idade_row.empty:
        idade_valor = idade_row['Idade (meses)'].values[0]  # extrai como escalar

        images = [Image.open(f) for f in files]
        arrays = [np.array(image).reshape(-1, 1) for image in images]

        shape = arrays[0].shape
        idade_array = np.full(shape, idade_valor)

        # Inclui idade_array junto das demais variáveis
        stacked_array = np.hstack(arrays + [idade_array])

        estimated = np.mean(model.predict(stacked_array))
        idade.loc[idade['Talhao'] == talhao_nome, 'ESTIMADO'] = estimated
    else:
        print(f"[ERRO] Talhão '{talhao_nome}' não encontrado na planilha.")
