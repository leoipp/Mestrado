import os.path

import numpy as np
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
predictions = []
for key, files in shapes.items():
    # Ler as imagens
    images = [Image.open(f) for f in files]
    # Converter as imagens para arrays numpy
    arrays = [np.array(image).reshape(-1, 1) for image in images]
    # Empilhar os arrays em um único array 3D
    stacked_array = np.hstack(arrays)
    print(stacked_array.shape)
    # estimated = model.predict(stacked_array)
    # predictions.append(estimated)
