import numpy as np
from PIL import Image
import joblib

#%% Load the Random Forest model
model = joblib.load(r'G:\PycharmProjects\Mestrado\Forecast\Predictive\Models\RandomForestRegressor.pkl')
print('Model variables:')
print(model.feature_names)

#%% Function to read TIFF immage and convert to np.ndarray
def read_tiff_image_as_array(image_path:str, reshape:bool=True):
    with Image.open(image_path) as img:
        img_array = np.array(img)
        if reshape:
            img_array = img_array.reshape(-1, 1)
        return img_array
