# 1. Library imports
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# 2. Load the encoder and scalers
RESOURCES_PATH = Path('resources')
if not RESOURCES_PATH.exists():
    raise FileNotFoundError(f"Resources directory not found at {RESOURCES_PATH}")

with (RESOURCES_PATH / 'encoder.pkl').open('rb') as f:
    encoder_load = pickle.load(f)

with (RESOURCES_PATH / 'scaler.pkl').open('rb') as f:
    scaler_load = pickle.load(f)

with (RESOURCES_PATH / 'scaler2.pkl').open('rb') as f:
    scaler2_load = pickle.load(f)

# 3. Define the numeric and categorical features
numeric_features = ['edad', 'dias', 'transaccion']
categorical_features = [
    'agencia_origen', 'linea', 'agencia_destino', 'tipo_cliente', 'codigo_actividad', 
    'sexo', 'estrato', 'tipo_identificacion', 'estudios', 'canal', 'medio_transaccion', 
    'tipo_entidad'
]

# 4. Function to transform user input
def func_transform(user_input: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler_load.transform(user_input[numeric_features])
    X_scaled = scaler2_load.transform(X_scaled)

    X_categorical = encoder_load.transform(user_input[categorical_features])
    X = np.hstack([X_categorical.toarray(), X_scaled])

    return X

# 5. Function to format the prediction output
def salida_pred(predicted: np.ndarray) -> str:
    return f'prediction es : {predicted[0]}'
