# 1. Library imports
from pathlib import Path
import uvicorn
from fastapi import FastAPI
import pandas as pd
import pickle
from typing import Dict
from inputs_data import Inputs
from my_functions import func_transform

# 2. Load the model and other resources
MODEL_PATH = Path('resources/clf.pkl')
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with MODEL_PATH.open('rb') as f:
    clf_load = pickle.load(f)

# 3. Create the app object
app = FastAPI()

# 4. Index route, opens automatically on http://127.0.0.0:8000
@app.get('/')
def index():
    return {'message': 'Welcome to level predict API'}

# 5. Predict route
@app.post('/predict')
def predict(data: Inputs) -> Dict[str, int]:
    data_dict = data.__dict__
    new_data = pd.DataFrame([data_dict])
    
    X = func_transform(new_data)
    out_model = clf_load.predict(X)
    
    return {"prediction": int(out_model[0])}

# 6. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
