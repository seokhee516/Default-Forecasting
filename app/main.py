import os
import sys

from typing import List
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse

import pandas as pd

app = FastAPI()

@app.post('/model/predict/')
def predict():
    """
    예측 결과를 반환하는 함수
    """
    output = "predict output complete"

    return output

@app.post('/model/shap/')
def calculate_shap_values():
   
    json_data = request.get_json()
    preprocessing_objects, model = helper.load_preprocessor_and_model()
    scaler = preprocessing_objects['scaler']
    features_selected = preprocessing_objects['features_selected']

    data = pd.DataFrame(json_data)[features_selected]
    preprocessed_data = helper.preprocess_record(data, scaler)
    base_value, shap_values = helper.get_base_and_shap_values(preprocessed_data, model)

    output = data.to_dict('records')[0]
    output['base_value'] = base_value
    output['shap_values'] = shap_values
    output = "shap output complete"

    return output