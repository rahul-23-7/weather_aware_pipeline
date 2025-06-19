# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:39:37 2025

@author: rahul
"""

# inference.py
import json
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import os

def model_fn(model_dir):
    """Load the model from the specified directory."""
    model_path = os.path.join(model_dir, 'random_forest_model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Process the input data."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        df = pd.DataFrame(data)
    elif request_content_type == 'text/csv':
        df = pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    # Ensure required features are present
    expected_features = [
        'temperature', 'precipitation', 'humidity',
        'temp_ma7', 'precip_ma7', 'sales_qty_lag1', 'sales_qty_lag7'
    ]
    if not all(col in df.columns for col in expected_features):
        raise ValueError(f"Missing required features: {expected_features}")
    
    return df[expected_features]

def predict_fn(input_data, model):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, accept):
    """Format the output."""
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()}), accept
    elif accept == 'text/csv':
        return pd.DataFrame({'predictions': prediction}).to_csv(index=False), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")